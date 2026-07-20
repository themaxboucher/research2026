import argparse
import logging
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from tqdm.auto import tqdm
from storage.datasets import dataset_directory_from_argument, latest_dataset_directory
from storage import iter_from_jsonl, save_to_jsonl

from collect.constants import DATA_FILENAME

SAMPLE_FILENAME = "dataset_sample"
DEFAULT_NUM_COMMITS = 100
DEFAULT_MAX_FILES_PER_COMMIT = 5


def _commit_key(record: dict) -> tuple[str, str]:
    return record["repo_name"], record["commit_hash"]


def _iter_commit_groups(records: Iterable[dict]) -> Iterator[list[dict]]:
    seen_keys: set[int] = set()
    current_key: tuple[str, str] | None = None
    current_group: list[dict] = []

    for record in records:
        key = _commit_key(record)
        if key != current_key:
            if current_group:
                yield current_group
            if current_key is not None:
                key_digest = hash(current_key)
                if key_digest in seen_keys:
                    logging.warning(
                        "Commit %s reappeared non-consecutively; its files may be "
                        "split across multiple groups",
                        current_key,
                    )
                seen_keys.add(key_digest)
            current_key = key
            current_group = []
        current_group.append(record)

    if current_group:
        yield current_group


def _reservoir_sample_commits(
    records: Iterable[dict],
    num_commits: int,
    max_files_per_commit: int,
    random_num_generator: random.Random,
) -> list[list[dict]]:
    # Reservoir sampling keeps a uniform random subset of qualifying commits in a
    # single pass, so we never hold the full dataset.jsonl (>100 GB) in memory.
    reservoir: list[list[dict]] = []
    qualifying_index = 0
    progress_bar = tqdm(records, desc="Sampling records", unit="record")
    for group in _iter_commit_groups(progress_bar):
        if len(group) > max_files_per_commit:
            continue

        if qualifying_index < num_commits:
            reservoir.append(group)
        else:
            candidate_index = random_num_generator.randint(0, qualifying_index)
            if candidate_index < num_commits:
                reservoir[candidate_index] = group
        qualifying_index += 1

    return reservoir


def sample_dataset(
    dataset_directory: Path,
    num_commits: int = DEFAULT_NUM_COMMITS,
    max_files_per_commit: int = DEFAULT_MAX_FILES_PER_COMMIT,
    seed: int | None = None,
) -> int:
    random_num_generator = random.Random(seed)
    records: Iterator[dict] = iter_from_jsonl(dataset_directory, DATA_FILENAME)
    sampled_commits = _reservoir_sample_commits(
        records, num_commits, max_files_per_commit, random_num_generator
    )

    if len(sampled_commits) < num_commits:
        logging.warning(
            "Only %d commits with <= %d files were available; requested %d",
            len(sampled_commits),
            max_files_per_commit,
            num_commits,
        )

    sample = [record for group in sampled_commits for record in group]
    save_to_jsonl(sample, dataset_directory, SAMPLE_FILENAME)
    logging.info(
        "Wrote %d files from %d commits to %s.jsonl",
        len(sample),
        len(sampled_commits),
        SAMPLE_FILENAME,
    )
    return len(sample)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-commits",
        type=int,
        default=DEFAULT_NUM_COMMITS,
        help="Number of commits to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--max-files-per-commit",
        type=int,
        default=DEFAULT_MAX_FILES_PER_COMMIT,
        help="Only sample commits with at most this many files (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible sampling",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to sample from (defaults to the latest dataset)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = (
        dataset_directory_from_argument(args.dataset_dir)
        if args.dataset_dir
        else latest_dataset_directory()
    )

    logging.info("Sampling from dataset directory: %s", dataset_directory)
    sample_dataset(
        dataset_directory,
        num_commits=args.num_commits,
        max_files_per_commit=args.max_files_per_commit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
