import argparse
import logging
from pathlib import Path

from generate.constants import SOURCE_FILENAME
from generate.providers.models import get_model_profile
from storage.runs import read_manifest, resolve_dataset_and_run, write_manifest


def _prepare(
    dataset_dir: Path,
    run_dir: Path,
    num_partitions: int | None,
    limit: int | None,
) -> tuple[int, int]:

    source_path = dataset_dir / f"{SOURCE_FILENAME}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"No file at {source_path}.")

    model_profile, model_profile_name = get_model_profile()

    existing_manifest = read_manifest(run_dir)

    if existing_manifest:
        existing_config = existing_manifest.get("config") or {}
        existing_num_partitions = existing_config.get("num_partitions")
        if existing_num_partitions is None:
            raise RuntimeError(
                f"Run {run_dir.name} was generated without a job array"
            )
        if num_partitions is not None and num_partitions != existing_num_partitions:
            raise ValueError(
                f"Run {run_dir.name} was created with --num-partitions "
                f"{existing_num_partitions}; resuming with {num_partitions} would "
                "reshuffle which task owns which record"
            )
        # A task's array index encodes which model it runs, so changing the
        # model list would reassign models to indices mid-run.
        existing_models = list(existing_manifest.get("model_names") or [])
        if existing_models != list(model_profile.model_names):
            raise ValueError(
                f"Run {run_dir.name} was created with models {existing_models}; "
                f"resuming with {list(model_profile.model_names)} would "
                "reshuffle which task runs which model"
            )
        num_partitions = existing_num_partitions
        limit = existing_config.get("max_generate")

    if num_partitions is None:
        raise ValueError("--num-partitions is required for a new array run")
    if num_partitions < 1:
        raise ValueError(f"--num-partitions must be >= 1, got {num_partitions}")

    write_manifest(
        run_dir,
        model_profile=model_profile_name,
        model_names=model_profile.model_names,
        config={
            "max_generate": limit,
            "num_partitions": num_partitions,
        },
        created_at=existing_manifest.get("created_at"),
    )
    # The job array covers every (model, partition) pair so each task loads a
    # single model and has the whole GPU allocation to itself.
    array_size = num_partitions * len(model_profile.model_names)
    return num_partitions, array_size


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
        "--max-generate",
        type=int,
        default=None,
        help="Limit records sent to the LLM for generation",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=None,
        help="Number of partitions to split the dataset into. The job array "
        "runs one task per (model, partition) pair",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory, run_directory = resolve_dataset_and_run(
        args.dataset_dir, args.run_dir, create_run=True
    )

    num_partitions, array_size = _prepare(
        dataset_dir=dataset_directory,
        run_dir=run_directory,
        num_partitions=args.num_partitions,
        limit=args.max_generate,
    )
    # submit.sh uses these prints to parse the array parameters
    print(f"DATASET_DIR={dataset_directory}")
    print(f"RUN_DIR={run_directory}")
    print(f"NUM_PARTITIONS={num_partitions}")
    print(f"ARRAY_SIZE={array_size}")


if __name__ == "__main__":
    main()
