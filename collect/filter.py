import argparse
import logging
from pathlib import Path

from tqdm.auto import tqdm

from collect.comments import get_comments_from_file
from collect.constants import DATASET_FILENAME, RAW_DATASET_FILENAME
from collect.filter_rules import (
    get_target_comments,
    has_eligible_metadata,
    is_ai_authored_file,
)
from collect.prompt import build_prompt
from collect.prompt_code import get_prompt_code
from storage import append_to_jsonl, iter_from_jsonl, save_to_jsonl
from storage.datasets import resolve_dataset_directory, write_manifest

WRITE_BATCH_SIZE = 1000


def _add_prompt_code_and_prompt(record: dict, target_comments: list[dict]) -> dict:
    source_code = record.get("source_code")
    if source_code is None:
        raise ValueError("source_code must be present in the record")

    for target_comment in target_comments:
        prompt_code = get_prompt_code(source_code, target_comment)
        target_comment["prompt_code"] = prompt_code
        target_comment["prompt"] = build_prompt(
            record["repo_name"],
            record["new_path"],
            target_comment,
            record["commit_message"],
            prompt_code,
        )

    record["target_comments"] = target_comments
    return record


def _filter_dataset(dataset_directory: Path) -> None:
    manifest = {
        "raw_num_repos": 0,
        "raw_num_commits": 0,
        "raw_num_files": 0,
        "num_files_wrong_metadata": 0,
        "num_files_ai_authored": 0,
        "num_files_comment_parse_error": 0,
        "num_files_no_target_comments": 0,
        "num_files_prompt_code_error": 0,
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

        if not has_eligible_metadata(record):
            manifest["num_files_wrong_metadata"] += 1
            continue

        if is_ai_authored_file(record):
            manifest["num_files_ai_authored"] += 1
            continue

        try:
            comments = get_comments_from_file(
                record["source_code"], record["previous_source_code"]
            )
            manifest["num_comments"] += len(comments)
        except Exception as e:
            logging.warning(
                "Failed to parse comments for %s: %s", record.get("new_path"), e
            )
            manifest["num_files_comment_parse_error"] += 1
            continue

        target_comments = get_target_comments(comments)
        if not target_comments:
            manifest["num_files_no_target_comments"] += 1
            continue

        try:
            record = _add_prompt_code_and_prompt(record, target_comments)
        except Exception as e:
            logging.warning(
                "Failed to add prompt code and prompt for %s: %s",
                record.get("new_path"),
                e,
            )
            manifest["num_files_prompt_code_error"] += 1
            continue

        manifest["num_target_comments"] += len(target_comments)
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
