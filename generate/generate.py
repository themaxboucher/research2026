import argparse
import itertools
import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from generate.approaches import (
    location_generate_for_file,
    regenerate_generate_for_file,
)
from generate.constants import (
    LOCATION_FILENAME,
    PROGRESS_FILENAME,
    REGENERATE_FILENAME,
    SOURCE_FILENAME,
)
from generate.providers.models import ModelProfile, get_model_profile
from storage import (
    append_to_jsonl,
    drop_trailing_records,
    iter_from_jsonl,
    save_to_jsonl,
    truncate_broken_tail,
)
from storage.runs import (
    MANIFEST_FILENAME,
    read_manifest,
    resolve_dataset_and_run,
)


def _file_key(record: dict) -> tuple[str | None, str | None, str | None]:
    return (record.get("repo_name"), record.get("new_path"), record.get("commit_hash"))


def _completed_file_keys(
    run_dir: Path, progress_filename: str
) -> set[tuple[str | None, str | None, str | None]]:
    if not (run_dir / f"{progress_filename}.jsonl").exists():
        return set()
    return {_file_key(record) for record in iter_from_jsonl(run_dir, progress_filename)}


def _shard_suffix(task_id: int, num_tasks: int) -> str:
    digit_width = max(len(str(num_tasks - 1)), 1)
    return f"{task_id:0{digit_width}d}"


def _shard_filename(filename: str, suffix: str | None) -> str:
    return f"{filename}.{suffix}" if suffix is not None else filename


def _output_filenames(approaches: list[str], suffix: str | None) -> list[str]:
    return [
        _shard_filename(filename, suffix)
        for filename, approach in (
            (LOCATION_FILENAME, "location"),
            (REGENERATE_FILENAME, "regenerate"),
        )
        if approach in approaches
    ]


def _generate_for_file(
    file_data: dict,
    model_profile: ModelProfile,
    approaches: list[str],
) -> list[tuple[str, list[dict]]]:
    outputs: list[tuple[str, list[dict]]] = []
    if "regenerate" in approaches:
        scope_records = regenerate_generate_for_file(file_data, model_profile)
        outputs.append((REGENERATE_FILENAME, scope_records))
    if "location" in approaches:
        location_record = location_generate_for_file(file_data, model_profile)
        outputs.append((LOCATION_FILENAME, [location_record]))
    return outputs


def _generate(dataset_dir: Path, run_dir: Path, task_id: int, config: dict) -> None:
    num_tasks = config["num_tasks"]
    approaches = config["approaches"]
    limit = config["max_generate"]
    model_profile, _ = get_model_profile()

    suffix = _shard_suffix(task_id, num_tasks)
    progress_filename = _shard_filename(PROGRESS_FILENAME, suffix)
    output_filenames = _output_filenames(approaches, suffix)

    concurrent_files = model_profile.concurrent_files

    # Resume an interrupted run rather than overwriting it
    truncate_broken_tail(run_dir, progress_filename)
    completed_keys = _completed_file_keys(run_dir, progress_filename)
    for filename in output_filenames:
        truncate_broken_tail(run_dir, filename)
        drop_trailing_records(
            run_dir,
            filename,
            lambda record: _file_key(record) not in completed_keys,
        )
    if completed_keys:
        logging.info(
            "Resuming run %r at %s (%d files already done).",
            run_dir.name,
            dataset_dir.name,
            len(completed_keys),
        )

    # Create empty output files for a fresh run
    for filename in output_filenames:
        if not (run_dir / f"{filename}.jsonl").exists():
            save_to_jsonl([], run_dir, filename)

    files_data = iter_from_jsonl(dataset_dir, SOURCE_FILENAME)

    if limit is not None:
        files_data = itertools.islice(files_data, limit)

    # Striding over the eligible files deterministically partitions files across tasks.
    files_data = itertools.islice(files_data, task_id, None, num_tasks)

    # Filter out files already completed by an earlier run of this generation. This
    # runs after `limit` so the target set matches a fresh run's first `limit`
    # eligible files. We just don't redo the ones already finished.
    files_data = (
        file_data
        for file_data in files_data
        if _file_key(file_data) not in completed_keys
    )

    # Count files already done so progress logging reflects the whole run.
    files_processed = len(completed_keys)

    # Files are generated concurrently but submitted through a window of at
    # most `concurrent_files`, so the source JSONL streams instead of being
    # held in memory. All appends happen here on the main thread.
    with ThreadPoolExecutor(max_workers=concurrent_files) as executor:
        in_flight: dict[Future, dict] = {}

        def submit_next_file() -> bool:
            file_data = next(files_data, None)
            if file_data is None:
                return False
            future = executor.submit(
                _generate_for_file, file_data, model_profile, approaches
            )
            in_flight[future] = {
                "repo_name": file_data.get("repo_name"),
                "new_path": file_data.get("new_path"),
                "commit_hash": file_data.get("commit_hash"),
            }
            return True

        for _ in range(concurrent_files):
            if not submit_next_file():
                break

        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                file_key = in_flight.pop(future)
                try:
                    outputs = future.result()
                except Exception as error:
                    # A failed file is recorded as done (with its error) so one
                    # pathological input can't wedge the task or spin forever
                    # on resume; it just produces no output records.
                    logging.warning(
                        "Generation failed for %s: %s", file_key["new_path"], error
                    )
                    append_to_jsonl(
                        [{**file_key, "error": str(error)}],
                        run_dir,
                        progress_filename,
                    )
                else:
                    for output_filename, records in outputs:
                        append_to_jsonl(
                            records, run_dir, _shard_filename(output_filename, suffix)
                        )
                    # Log only after every approach's output is on disk, so a
                    # crash mid-write leaves this file uncommitted and it is redone.
                    append_to_jsonl(
                        [{**file_key, "error": None}], run_dir, progress_filename
                    )
                    logging.info(
                        "Generated comments for %s (file %d)",
                        file_key["new_path"],
                        files_processed + 1,
                    )
                files_processed += 1
                submit_next_file()

    logging.info(
        "Finished generation %r at %s (%d files generated).",
        run_dir.name,
        dataset_dir.name,
        files_processed,
    )


def _valid_manifest_config(run_dir: Path, task_id: int) -> dict:
    manifest = read_manifest(run_dir)
    if not manifest:
        raise RuntimeError(f"No manifest at {run_dir / MANIFEST_FILENAME}.")

    config = manifest.get("config") or {}

    num_tasks = config.get("num_tasks")
    if num_tasks is None:
        raise RuntimeError(f"Run {run_dir.name} was not prepared for a job array")

    approaches = config.get("approaches")
    if approaches is None:
        raise RuntimeError(
            f"Run {run_dir.name} has no generation approaches configured"
        )

    invalid_task_id = not 0 <= task_id < num_tasks
    if invalid_task_id:
        raise ValueError(
            f"--task-id {task_id} is out of range for --num-tasks {num_tasks}"
        )

    _, model_profile_name = get_model_profile()
    manifest_profile = manifest.get("model_profile")
    if manifest_profile is not None and manifest_profile != model_profile_name:
        raise RuntimeError(
            f"Run {run_dir.name} was prepared with MODEL_PROFILE="
            f"{manifest_profile!r} but this task is running with "
            f"{model_profile_name!r}"
        )

    return config


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to generate for (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to generate for (defaults to the latest run in the dataset)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the job array. Generates only this task's "
        "share of the eligible files into its own sharded files",
    )

    args = parser.parse_args()

    if args.task_id is None:
        raise SystemExit(
            "--task-id is required: it is this task's index in the job array"
        )

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_dir, run_dir = resolve_dataset_and_run(args.dataset_dir, args.run_dir)

    manifest_config = _valid_manifest_config(run_dir, args.task_id)

    return _generate(
        dataset_dir,
        run_dir,
        task_id=args.task_id,
        config=manifest_config,
    )


if __name__ == "__main__":
    main()
