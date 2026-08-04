import argparse
import logging
from pathlib import Path

from eval.manifest import read_eval_manifest, write_eval_manifest
from generate.constants import LOCATION_FILENAME, REGENERATE_FILENAME
from storage.runs import resolve_dataset_and_run


def _prepare(run_dir: Path, num_tasks: int | None) -> int:
    has_output = any(
        (run_dir / f"{filename}.jsonl").exists()
        for filename in (LOCATION_FILENAME, REGENERATE_FILENAME)
    )
    if not has_output:
        raise SystemExit(f"No generation output to score in {run_dir}")

    existing = read_eval_manifest(run_dir)
    if existing:
        existing_num_tasks = existing.get("num_tasks")
        if num_tasks is not None and num_tasks != existing_num_tasks:
            raise ValueError(
                f"Run {run_dir.name} was prepared with --num-tasks "
                f"{existing_num_tasks}; resuming with {num_tasks} would "
                "reshuffle which task owns which record"
            )
        num_tasks = existing_num_tasks

    if num_tasks is None:
        raise ValueError("--num-tasks is required for a new array run")
    if num_tasks < 1:
        raise ValueError(f"--num-tasks must be >= 1, got {num_tasks}")

    write_eval_manifest(run_dir, num_tasks=num_tasks)
    return num_tasks


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to evaluate for (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to evaluate (defaults to the latest run in the dataset)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the scoring job array"
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory, run_directory = resolve_dataset_and_run(
        args.dataset_dir, args.run_dir
    )

    num_tasks = _prepare(run_directory, args.num_tasks)

    # submit.sh uses these prints to parse the array parameters
    print(f"DATASET_DIR={dataset_directory}")
    print(f"RUN_DIR={run_directory}")
    print(f"NUM_TASKS={num_tasks}")


if __name__ == "__main__":
    main()
