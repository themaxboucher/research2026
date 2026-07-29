import argparse
import logging
from pathlib import Path

from tqdm.auto import tqdm

from collect.comments import get_comments_from_file
from collect.constants import DATASET_FILENAME, RAW_DATASET_FILENAME
from collect.filter_rules import (
    has_eligible_metadata,
    target_comments,
)
from storage import append_to_jsonl, iter_from_jsonl, save_to_jsonl
from storage.datasets import resolve_dataset_directory, write_manifest

WRITE_BATCH_SIZE = 1000


def _add_parsed_comments(file_data):
    file_data["comments"] = []
    source_code = file_data.get("source_code")
    previous_source_code = file_data.get("previous_source_code")

    if source_code is None or previous_source_code is None:
        return file_data

    try:
        file_data["comments"] = get_comments_from_file(
            source_code, previous_source_code
        )
    except Exception as e:
        logging.warning(
            "Failed to parse comments for %s: %s", file_data.get("new_path"), e
        )
        file_data["error"] = str(e)
    return file_data


def _filter_dataset(dataset_directory: Path) -> None:
    manifest = {
        "raw_num_repos": 0,
        "raw_num_commits": 0,
        "raw_num_files": 0,
        "num_repos": 0,
        "num_commits": 0,
        "num_files": 0,
        "num_comments": 0,
        "num_target_comments": 0,
    }

    counted_raw_repos: set[str] = set()
    counted_raw_commits: set[tuple[str, str]] = set()
    counted_repos: set[str] = set()
    counted_commits: set[tuple[str, str]] = set()

    save_to_jsonl([], dataset_directory, DATASET_FILENAME)
    kept_records: list[dict] = []

    records = iter_from_jsonl(dataset_directory, RAW_DATASET_FILENAME)

    for record in tqdm(records, desc="Filtering records", unit="record"):
        manifest["raw_num_files"] += 1

        repo_key = record["repo_name"]
        if repo_key not in counted_raw_repos:
            manifest["raw_num_repos"] += 1
            counted_raw_repos.add(repo_key)

        commit_key = (record["repo_name"], record["commit_hash"])
        if commit_key not in counted_raw_commits:
            manifest["raw_num_commits"] += 1
            counted_raw_commits.add(commit_key)

        # Reject records ruled out from metadata alone before parsing, so
        # non-Python files aren't tokenized as Python only to be discarded.
        if not has_eligible_metadata(record):
            continue

        record = _add_parsed_comments(record)
        manifest["num_comments"] += len(record["comments"])

        targets = target_comments(record)
        if not targets:
            continue

        manifest["num_target_comments"] += len(targets)
        manifest["num_files"] += 1

        if repo_key not in counted_repos:
            manifest["num_repos"] += 1
            counted_repos.add(repo_key)

        if commit_key not in counted_commits:
            manifest["num_commits"] += 1
            counted_commits.add(commit_key)

        kept_records.append(record)
        if len(kept_records) >= WRITE_BATCH_SIZE:
            append_to_jsonl(kept_records, dataset_directory, DATASET_FILENAME)
            kept_records.clear()

    if kept_records:
        append_to_jsonl(kept_records, dataset_directory, DATASET_FILENAME)

    write_manifest(dataset_directory, manifest)
    logging.info(
        "Kept %d of %d files (%d of %d commits, %d of %d repos) with %d target comments",
        manifest["num_files"],
        manifest["raw_num_files"],
        manifest["num_commits"],
        manifest["raw_num_commits"],
        manifest["num_repos"],
        manifest["raw_num_repos"],
        manifest["num_target_comments"],
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to sample from (defaults to the latest dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = resolve_dataset_directory(args.dataset_dir)

    _filter_dataset(dataset_directory)


if __name__ == "__main__":
    main()
