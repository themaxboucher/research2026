from runs import require_latest_run_directory
from storage import iter_from_jsonl, save_to_jsonl
from collections.abc import Iterable, Iterator
from pathlib import Path
import argparse
import logging
import random

DATA_FILENAME = "repo_files"
SAMPLE_FILENAME = "repo_files_sample"
DEFAULT_SAMPLE_SIZE = 100


def _reservoir_sample(
    records: Iterable[dict], sample_size: int, random_num_generator: random.Random
) -> list[dict]:
    # Reservoir sampling keeps a uniform random subset in a single pass, so we
    # never hold the full repo_files.jsonl (>100 GB) in memory.
    reservoir: list[dict] = []
    for index, record in enumerate(records):
        if index < sample_size:
            reservoir.append(record)
            continue

        candidate_index = random_num_generator.randint(0, index)
        if candidate_index < sample_size:
            reservoir[candidate_index] = record

    return reservoir


def sample_dataset(
    run_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
) -> int:
    random_num_generator = random.Random(seed)
    records: Iterator[dict] = iter_from_jsonl(run_dir, DATA_FILENAME)
    sample = _reservoir_sample(records, sample_size, random_num_generator)

    save_to_jsonl(sample, run_dir, SAMPLE_FILENAME)
    logging.info("Wrote %d sampled records to %s.jsonl", len(sample), SAMPLE_FILENAME)
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample entries from repo_files.jsonl"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of entries to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible sampling",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to sample from (defaults to the latest run)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else require_latest_run_directory()
    logging.info("Sampling from run directory: %s", run_dir)
    sample_dataset(run_dir, sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
