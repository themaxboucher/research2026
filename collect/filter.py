import argparse
import logging
import zlib
from pathlib import Path

from tqdm.auto import tqdm

from collect.comments import get_comments_from_change
from collect.constants import DATASET_FILENAME, RAW_DATASET_FILENAME
from collect.filter_rules import (
    get_target_comments,
    has_eligible_metadata,
    is_ai_authored_file,
)
from collect.prompt import build_prompt
from collect.prompt_code import get_prompt_code
from storage import (
    append_to_jsonl,
    iter_from_jsonl,
    save_to_jsonl,
    shard_filename,
    shard_suffix,
)
from storage.datasets import (
    MANIFEST_FILENAME,
    resolve_dataset_directory,
    write_manifest,
)

WRITE_BATCH_SIZE = 1000


def _task_owns_commit(
    repo_name: str, commit_hash: str, task_id: int, num_tasks: int
) -> bool:
    """Partition the raw dataset by commit, so files from a commit are not split across tasks."""
    partition_key = f"{repo_name}@{commit_hash}".encode("utf-8")
    return zlib.crc32(partition_key) % num_tasks == task_id


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


def _filter_dataset(
    dataset_directory: Path,
    task_id: int | None = None,
    num_tasks: int | None = None,
) -> None:
    in_jobs_array = task_id is not None and num_tasks is not None
    suffix = shard_suffix(task_id, num_tasks) if in_jobs_array else None
    dataset_filename = shard_filename(DATASET_FILENAME, suffix)
    manifest_filename = shard_filename(MANIFEST_FILENAME, suffix)

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

    save_to_jsonl([], dataset_directory, dataset_filename)
    kept_records: list[dict] = []

    records = iter_from_jsonl(dataset_directory, RAW_DATASET_FILENAME)

    for record in tqdm(records, desc="Filtering records", unit="record"):
        if in_jobs_array and not _task_owns_commit(
            record["repo_name"], record["commit_hash"], task_id, num_tasks
        ):
            continue

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
            comments = get_comments_from_change(
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
            append_to_jsonl(kept_records, dataset_directory, dataset_filename)
            kept_records.clear()

    if kept_records:
        append_to_jsonl(kept_records, dataset_directory, dataset_filename)

    if in_jobs_array:
        write_manifest(
            dataset_directory,
            {
                "task_id": task_id,
                "num_tasks": num_tasks,
                "counts": manifest,
                "raw_repo_names": sorted(counted_raw_repos),
                "repo_names": sorted(counted_repos),
            },
            manifest_filename,
        )
    else:
        write_manifest(dataset_directory, manifest)

    logging.info(
        "Kept %d of %d files (%d target comments) across %d repos",
        manifest["num_files"],
        manifest["raw_num_files"],
        manifest["num_target_comments"],
        manifest["num_repos"],
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to sample from (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the filtering job array. Filters only the "
        "repos owned by this task into its own sharded files",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the filtering job array. Filters only the "
        "repos owned by this task into its own sharded files",
    )
    args = parser.parse_args()

    if (args.task_id is None) != (args.num_tasks is None):
        raise SystemExit(
            "--task-id and --num-tasks must be given together: both partition "
            "the commits, and one without the other cannot"
        )
    if args.num_tasks is not None:
        if args.num_tasks < 1:
            raise SystemExit(f"--num-tasks must be >= 1, got {args.num_tasks}")
        if not 0 <= args.task_id < args.num_tasks:
            raise SystemExit(
                f"--task-id {args.task_id} is out of range for "
                f"--num-tasks {args.num_tasks}"
            )

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = resolve_dataset_directory(args.dataset_dir)

    _filter_dataset(
        dataset_directory,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()
