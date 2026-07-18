import argparse
import logging
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from pydriller import Repository
from datetime import datetime

from collect.prepare import get_repos
from collect.dataset import (
    dataset_directory_from_argument,
    dataset_directory_timestamp,
    resolve_dataset_directory,
)
from storage import (
    append_to_jsonl,
    drop_trailing_records,
    load_from_jsonl,
    truncate_broken_tail,
)
from collect.constants import (
    DATA_FILENAME,
    MINNED_REPOS_FILENAME,
    CUTOFF_DATE,
    DEFAULT_MAX_REPOS,
)


def _get_shard_filenames(task_id: int, num_tasks: int) -> tuple[str, str]:
    digit_width = max(len(str(num_tasks - 1)), 1)
    formatted_suffix = f"{task_id:0{digit_width}d}"
    return (
        f"{DATA_FILENAME}.{formatted_suffix}",
        f"{MINNED_REPOS_FILENAME}.{formatted_suffix}",
    )


def _clean_previous_data(
    run_dir: Path, data_filename: str, mined_filename: str
) -> None:
    truncate_broken_tail(run_dir, mined_filename)
    removed_bytes = truncate_broken_tail(run_dir, data_filename)
    if removed_bytes:
        logging.warning(
            "Removed %d bytes of partially written data from %s.jsonl",
            removed_bytes,
            data_filename,
        )

    mined_repo_names = _mined_repo_names(run_dir, mined_filename)
    removed_records = drop_trailing_records(
        run_dir,
        data_filename,
        lambda record: record["repo_name"] not in mined_repo_names,
    )
    if removed_records:
        logging.warning(
            "Removed %d records from an interrupted repo. It will be re-mined",
            removed_records,
        )


def _mined_repo_names(run_dir: Path, mined_filename: str) -> set[str]:
    mined_repos_path = run_dir / f"{mined_filename}.jsonl"
    if not mined_repos_path.exists():
        return set()
    mined_repos = load_from_jsonl(run_dir, mined_filename)
    return {repo["repo"] for repo in mined_repos}


def _unmined_repos(repos: list[dict], run_dir: Path, mined_filename: str) -> list[dict]:
    mined_repo_names = _mined_repo_names(run_dir, mined_filename)
    return [repo for repo in repos if repo["full_name"] not in mined_repo_names]


def _sort_repos_by_size(repos: list[dict]) -> list[dict]:
    return sorted(repos, key=lambda repo: repo["size"])


def _mine_repo(
    repo_url: str, repo_full_name: str, branch: str, since: str, to: datetime
) -> list[dict]:
    datetime_since = datetime.strptime(since, "%Y-%m-%d")

    repo = Repository(
        repo_url,
        since=datetime_since,
        to=to,
        only_in_branch=branch,
        only_no_merge=True,
        only_modifications_with_file_types=[".py"],
    )

    repo_files = []

    for commit in repo.traverse_commits():
        for file in commit.modified_files:
            is_python_file = file.filename.endswith(".py")
            if not is_python_file:
                continue

            repo_files.append(
                {
                    "repo_name": repo_full_name,
                    "commit_hash": commit.hash,
                    "commit_message": commit.msg,
                    "filename": file.filename,
                    "new_path": file.new_path,
                    "change_type": file.change_type.name,
                    "diff": file.diff,
                    "diff_parsed": file.diff_parsed,
                    "added_lines": file.added_lines,
                    "deleted_lines": file.deleted_lines,
                    "source_code": file.source_code,
                    "previous_source_code": file.source_code_before,
                    "nloc": file.nloc,
                    "complexity": file.complexity,
                    "token_count": file.token_count,
                }
            )

    return repo_files


MINING_MAX_ATTEMPTS = 3
MINING_RETRY_DELAY_SECONDS = 3


def _mine_repo_with_retries(
    repo_url: str, repo_full_name: str, branch: str, since: str, to: datetime
) -> list[dict]:
    for attempt in range(1, MINING_MAX_ATTEMPTS + 1):
        try:
            return _mine_repo(repo_url, repo_full_name, branch, since, to)
        except Exception as error:
            if attempt == MINING_MAX_ATTEMPTS:
                raise
            logging.warning(
                "Error mining %s (attempt %d/%d), retrying in %ds: %s",
                repo_url,
                attempt,
                MINING_MAX_ATTEMPTS,
                MINING_RETRY_DELAY_SECONDS,
                error,
            )
            time.sleep(MINING_RETRY_DELAY_SECONDS)


def _mine_and_persist_repo(
    repo: dict,
    run_dir: Path,
    write_lock: Lock,
    data_filename: str,
    mined_filename: str,
    mining_end: datetime,
) -> int:
    repo_url = repo["html_url"]
    try:
        repo_files = _mine_repo_with_retries(
            repo_url,
            repo["full_name"],
            repo["default_branch"],
            CUTOFF_DATE,
            mining_end,
        )
    except Exception as error:
        with write_lock:
            append_to_jsonl(
                [{"repo": repo["full_name"], "error": str(error)}],
                run_dir,
                mined_filename,
            )
        logging.warning("Failed to mine %s: %s", repo_url, error)
        return 0

    with write_lock:
        append_to_jsonl(repo_files, run_dir, data_filename)
        append_to_jsonl(
            [{"repo": repo["full_name"], "error": None}], run_dir, mined_filename
        )
    return len(repo_files)


def _collect(
    dataset_dir: Path,
    task_id: int | None = None,
    num_tasks: int | None = None,
    max_repos: int | None = None,
    repo_min_stars: int = 0,
) -> None:
    in_jobs_array = task_id is not None and num_tasks is not None
    if in_jobs_array:
        data_filename, mined_filename = _get_shard_filenames(task_id, num_tasks)
    else:
        data_filename, mined_filename = DATA_FILENAME, MINNED_REPOS_FILENAME

    all_repos = get_repos(repo_min_stars, max_repos, dataset_dir)
    mining_end = dataset_directory_timestamp(dataset_dir)

    sorted_repos = _sort_repos_by_size(all_repos)

    if in_jobs_array:
        # Ensure a balanced distribution of repo sizes across tasks
        repos_partition = sorted_repos[task_id::num_tasks]
    else:
        repos_partition = sorted_repos

    _clean_previous_data(dataset_dir, data_filename, mined_filename)

    repos = _unmined_repos(repos_partition, dataset_dir, mined_filename)

    if not repos:
        logging.info("No repositories left to mine for this task")
        return

    write_lock = Lock()
    MINING_WORKERS = 16
    workers = max(1, min(MINING_WORKERS, len(repos)))

    logging.info("Mining %d repositories with %d workers", len(repos), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _mine_and_persist_repo,
                repo,
                dataset_dir,
                write_lock,
                data_filename,
                mined_filename,
                mining_end,
            )
            for repo in repos
        ]
        progress_bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Mining repositories",
            unit="repo",
            leave=True,
        )
        for completed_future in progress_bar:
            completed_future.result()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to use instead of the latest one: a timestamp name "
        "(resolved under datasets/) or a path ending in a timestamp",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the job array. Mines only this task's share "
        "of the repos into its own sharded files",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the job array. Mines only this task's share "
        "of the repos into its own sharded files",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=DEFAULT_MAX_REPOS,
        help="Limit number of repos processed",
    )
    parser.add_argument(
        "--repo-min-stars",
        type=int,
        default=0,
        help="Only include repos with at least this many stars",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    if args.dataset_dir:
        dataset_directory = dataset_directory_from_argument(args.dataset_dir)
    else:
        dataset_directory = resolve_dataset_directory()

    _collect(
        dataset_directory,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
        max_repos=args.max_repos,
        repo_min_stars=args.repo_min_stars,
    )


if __name__ == "__main__":
    main()
