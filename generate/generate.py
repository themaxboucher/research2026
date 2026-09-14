import argparse
import itertools
import logging
from pathlib import Path
from typing import Callable

from generate.audit import comment_generation_succeeded
from generate.constants import (
    GENERATE_FILENAME,
    PROGRESS_FILENAME,
    SOURCE_FILENAME,
)
from generate.model_output import strip_output_wrappers
from generate.providers.models import get_model_profile
from generate.shards import dataset_record_key, repair_interrupted_shard
from storage import (
    append_to_jsonl,
    iter_from_jsonl,
    rewrite_jsonl,
    save_to_jsonl,
    shard_filename,
    shard_suffix,
)
from storage.runs import (
    MANIFEST_FILENAME,
    read_manifest,
    resolve_dataset_and_run,
)


def _task_assignment(
    task_id: int, num_partitions: int, model_names: list[str]
) -> tuple[str, int]:
    model_index, partition = divmod(task_id, num_partitions)
    return model_names[model_index], partition


def _generate_with_llm(
    prompt: str,
    filepath: str,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> list[dict]:
    raw_response = None
    try:
        raw_response = get_completion(model_name, prompt)
        comment_text = strip_output_wrappers(raw_response)
        if not comment_text:
            raise ValueError("Model returned an empty comment")
    except Exception as error:
        logging.warning(
            "Failed to generate comment in %s with model %s: %s",
            filepath,
            model_name,
            error,
        )
        return [
            {
                "model": model_name,
                "raw_response": raw_response,
                "comment_text": None,
                "error": str(error),
            }
        ]
    return [
        {
            "model": model_name,
            "raw_response": raw_response,
            "comment_text": comment_text,
            "error": None,
        }
    ]


def _generate_for_comment(
    dataset_record: dict,
    comment_data: dict,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict:
    prompt = comment_data.get("prompt")
    if prompt is None:
        raise ValueError("Comment data must include a prompt")

    filepath = dataset_record["new_path"]

    results = _generate_with_llm(prompt, filepath, model_name, get_completion)
    return {
        "type": comment_data["type"],
        "status": comment_data["status"],
        "start_line": comment_data["start_line"],
        "end_line": comment_data["end_line"],
        "anchor": comment_data.get("anchor"),
        "comment": comment_data.get("comment"),
        "prompt_code": comment_data.get("prompt_code"),
        "prompt": prompt,
        "results": results,
    }


def _generate_for_record(
    dataset_record: dict, model_name: str, get_completion: Callable[[str, str], str]
) -> dict:
    comment_generations = []
    for comment_data in dataset_record.get("target_comments", []):
        try:
            comment_generations.append(
                _generate_for_comment(
                    dataset_record, comment_data, model_name, get_completion
                )
            )
        except Exception as error:
            logging.warning(
                "Skipping a comment in %s: could not build generation inputs: %s",
                dataset_record.get("new_path"),
                error,
            )

    return {
        "repo_name": dataset_record["repo_name"],
        "commit_hash": dataset_record.get("commit_hash"),
        "new_path": dataset_record["new_path"],
        "model": model_name,
        "comment_generations": comment_generations,
    }


def _retry_failed_generations(
    generation_record: dict,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict:
    failed_generations = [
        comment_generation
        for comment_generation in generation_record.get("comment_generations") or []
        if not comment_generation_succeeded(comment_generation)
    ]
    if not failed_generations:
        return generation_record

    for failed_generation in failed_generations:
        failed_generation["results"] = _generate_with_llm(
            failed_generation["prompt"],
            generation_record["new_path"],
            model_name,
            get_completion,
        )
    logging.info(
        "Retried %d failed comments for %s with %s",
        len(failed_generations),
        generation_record["new_path"],
        model_name,
    )
    return generation_record


def _generate(
    dataset_dir: Path,
    run_dir: Path,
    task_id: int,
    config: dict,
    model_names: list[str],
    retry_failed: bool,
) -> None:
    num_partitions = config["num_partitions"]
    limit = config["max_generate"]
    model_profile, _ = get_model_profile()

    array_size = num_partitions * len(model_names)
    model_name, partition = _task_assignment(task_id, num_partitions, model_names)

    # The suffix is the flat array index, so every (model, partition) pair gets
    # its own output and progress shards and finalize merges them all.
    suffix = shard_suffix(task_id, array_size)
    progress_filename = shard_filename(PROGRESS_FILENAME, suffix)
    output_filename = shard_filename(GENERATE_FILENAME, suffix)

    logging.info(
        "Task %d generates with %s over partition %d of %d",
        task_id,
        model_name,
        partition,
        num_partitions,
    )

    # Resume an interrupted run rather than overwriting it
    completed_keys = repair_interrupted_shard(run_dir, suffix)
    if completed_keys:
        logging.info(
            "Resuming run %r at %s (%d records already done for %s).",
            run_dir.name,
            dataset_dir.name,
            len(completed_keys),
            model_name,
        )

    # Create empty output file for a fresh run
    if not (run_dir / f"{output_filename}.jsonl").exists():
        save_to_jsonl([], run_dir, output_filename)

    # Failed comments are regenerated in place, so each record stays a single
    # line in the shard and its progress row is still accurate.
    if retry_failed:
        rewrite_jsonl(
            run_dir,
            output_filename,
            lambda generation_record: _retry_failed_generations(
                generation_record, model_name, model_profile.get_completion
            ),
        )

    dataset_records = iter_from_jsonl(dataset_dir, SOURCE_FILENAME)

    if limit is not None:
        dataset_records = itertools.islice(dataset_records, limit)

    # Striding over the eligible records deterministically partitions them. Every
    # model walks the same partitions, so a record's generations all come from
    # the same stride regardless of which model ran it.
    dataset_records = itertools.islice(dataset_records, partition, None, num_partitions)

    # Filter out records already completed by an earlier run of this generation.
    # This runs after `limit` so the target set matches a fresh run's first
    # `limit` eligible records. We just don't redo the ones already finished.
    dataset_records = (
        dataset_record
        for dataset_record in dataset_records
        if dataset_record_key(dataset_record) not in completed_keys
    )

    # Count records already done so progress logging reflects the whole task.
    records_processed = len(completed_keys)

    for dataset_record in dataset_records:
        record_key = {
            "repo_name": dataset_record.get("repo_name"),
            "new_path": dataset_record.get("new_path"),
            "commit_hash": dataset_record.get("commit_hash"),
            "model": model_name,
        }
        try:
            generation_record = _generate_for_record(
                dataset_record, model_name, model_profile.get_completion
            )
        except Exception as error:
            # A failed record is recorded as done (with its error)
            logging.warning(
                "Generation failed for %s: %s", record_key["new_path"], error
            )
            append_to_jsonl(
                [{**record_key, "error": str(error)}], run_dir, progress_filename
            )
        else:
            append_to_jsonl([generation_record], run_dir, output_filename)
            # Log only after the output record is on disk, so a crash mid-write
            # leaves this record uncommitted and it is redone.
            append_to_jsonl([record_key], run_dir, progress_filename)
            logging.info(
                "Generated comments for %s with %s (record %d)",
                record_key["new_path"],
                model_name,
                records_processed + 1,
            )
        records_processed += 1

    logging.info(
        "Finished generation %r at %s (%d records generated with %s).",
        run_dir.name,
        dataset_dir.name,
        records_processed,
        model_name,
    )


def _valid_manifest_config(run_dir: Path, task_id: int) -> tuple[dict, list[str]]:
    manifest = read_manifest(run_dir)
    if not manifest:
        raise RuntimeError(f"No manifest at {run_dir / MANIFEST_FILENAME}.")

    config = manifest.get("config") or {}

    num_partitions = config.get("num_partitions")
    if num_partitions is None:
        raise RuntimeError(f"Run {run_dir.name} was not prepared for a job array")

    model_profile, model_profile_name = get_model_profile()
    manifest_profile = manifest.get("model_profile")
    if manifest_profile is not None and manifest_profile != model_profile_name:
        raise RuntimeError(
            f"Run {run_dir.name} was prepared with MODEL_PROFILE="
            f"{manifest_profile!r} but this task is running with "
            f"{model_profile_name!r}"
        )

    # The array index encodes which model a task runs, so the model list has to
    # be the one the run was prepared with or tasks would silently change model.
    model_names = manifest.get("model_names")
    if model_names is None:
        raise RuntimeError(f"Run {run_dir.name} has no models recorded")
    if list(model_names) != list(model_profile.model_names):
        raise RuntimeError(
            f"Run {run_dir.name} was prepared with models {list(model_names)} "
            f"but this task's profile defines {list(model_profile.model_names)}"
        )

    array_size = num_partitions * len(model_names)
    invalid_task_id = not 0 <= task_id < array_size
    if invalid_task_id:
        raise ValueError(
            f"--task-id {task_id} is out of range for an array of {array_size} "
            f"({num_partitions} partitions x {len(model_names)} models)"
        )

    return config, list(model_names)


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
        help="This task's index in the job array. Generates one model's "
        "comments for this task's partition of the eligible records into its "
        "own output shards",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Before resuming, regenerate the comments whose generation failed "
        "in this task's shard and replace their results in place. Successful "
        "comments are kept",
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

    manifest_config, model_names = _valid_manifest_config(run_dir, args.task_id)

    return _generate(
        dataset_dir,
        run_dir,
        task_id=args.task_id,
        config=manifest_config,
        model_names=model_names,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
