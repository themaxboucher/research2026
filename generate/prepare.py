import argparse
import logging
from pathlib import Path

from generate.models import get_model_profile
from generate.constants import SOURCE_FILENAME
from generate.runs import resolve_dataset_and_run, read_manifest, write_manifest
from generate.approaches import approaches_from_argument, APPROACHES


def _prepare(
    dataset_dir: Path,
    run_dir: Path,
    num_tasks: int | None,
    limit: int | None,
    approaches: list[str] | None,
) -> int:

    source_path = dataset_dir / f"{SOURCE_FILENAME}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"No sample file at {source_path}.")

    model_profile, model_profile_name = get_model_profile()

    existing_manifest = read_manifest(run_dir)

    if existing_manifest:
        existing_config = existing_manifest.get("config") or {}
        existing_num_tasks = existing_config.get("num_tasks")
        if existing_num_tasks is None:
            raise RuntimeError(
                f"Run {run_dir.name} was generated without a job array"
            )
        if num_tasks is not None and num_tasks != existing_num_tasks:
            raise ValueError(
                f"Run {run_dir.name} was created with --num-tasks "
                f"{existing_num_tasks}; resuming with {num_tasks} would "
                "reshuffle which task owns which file"
            )
        num_tasks = existing_num_tasks
        approaches = existing_config.get("approaches") or approaches
        limit = existing_config.get("max_generate")

    if num_tasks is None:
        raise ValueError("--num-tasks is required for a new array run")
    if num_tasks < 1:
        raise ValueError(f"--num-tasks must be >= 1, got {num_tasks}")

    write_manifest(
        run_dir,
        model_profile=model_profile_name,
        model_names=model_profile.model_names,
        config={
            "max_generate": limit,
            "approaches": approaches,
            "num_tasks": num_tasks,
        },
        created_at=existing_manifest.get("created_at"),
    )
    return num_tasks


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
        help="Run directory to store LLM outputs (defaults to a new timestamped directory)",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default=",".join(APPROACHES),
        help="Comma-separated generation approaches to run: 'location' prompts "
        "for a comment at a given spot, 'regenerate' has the model rewrite each "
        "scope with comments added (default: both)",
    )
    parser.add_argument(
        "--max-generate",
        type=int,
        default=None,
        help="Limit files sent to the LLM for generation",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the job array",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory, run_directory = resolve_dataset_and_run(
        args.dataset_dir, args.run_dir, create_run=True
    )

    approaches = approaches_from_argument(args.approaches)

    num_tasks = _prepare(
        dataset_dir=dataset_directory,
        run_dir=run_directory,
        num_tasks=args.num_tasks,
        limit=args.max_generate,
        approaches=approaches,
    )
    # submit.sh uses these prints to parse the array parameters
    print(f"DATASET_DIR={dataset_directory}")
    print(f"RUN_DIR={run_directory}")
    print(f"NUM_TASKS={num_tasks}")


if __name__ == "__main__":
    main()
