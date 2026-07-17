from collect.github import search_repos
from collect.dataset import dataset_directory_timestamp, resolve_dataset_directory
import argparse
from storage import (
    append_to_jsonl,
    drop_trailing_records,
    load_from_jsonl,
    merge_jsonl_shards,
    truncate_broken_tail,
)
from tqdm.auto import tqdm
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from pydriller import Repository
from datetime import datetime

# === LLM knowledge cutoffs ===
# GPT-5.6 Luna (https://developers.openai.com/api/docs/models/gpt-5.6-luna): Feb 16, 2026
# Llama 3.1 8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# Qwen 2.5 7b (https://huggingface.co/Qwen/Qwen2.5-7B): Unknown, but we know the model was released in Sep 2024

CUTOFF_DATE = "2026-02-17" # After GPT-5.6 Luna's knowledge cutoff, to avoid data leakage

REPO_LANGUAGE = "Python"
DEFAULT_MINING_WORKERS = 16

DATA_FILENAME = "dataset"
REPOS_CACHE_FILENAME = "repos_cache"
MINNED_REPOS_FILENAME = "mined_repos"


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


def _get_repos(repo_min_stars: int, max_repos: int | None, run_dir: Path) -> list[dict]:
    repos_cache_path = run_dir / f"{REPOS_CACHE_FILENAME}.jsonl"
    if repos_cache_path.exists():
        logging.info("Loading cached repositories from %s", repos_cache_path)
        return load_from_jsonl(run_dir, REPOS_CACHE_FILENAME)

    logging.info("Searching for repositories with at least %d stars", repo_min_stars)
    repos = search_repos(
        language=REPO_LANGUAGE,
        min_stars=repo_min_stars,
        pushed_after=CUTOFF_DATE,
        limit=max_repos,
    )[:max_repos]

    append_to_jsonl(repos, run_dir, REPOS_CACHE_FILENAME)

    return repos


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
        repo_files = _mine_repo(
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


def prepare_collection(
    run_dir: Path,
    max_repos: int | None,
    repo_min_stars: int,
    repos_per_task: int,
) -> int:
    repos = _get_repos(repo_min_stars, max_repos, run_dir)
    num_tasks = max(1, math.ceil(len(repos) / repos_per_task))
    logging.info(
        "Prepared %d repositories into %d tasks (<=%d repos each)",
        len(repos),
        num_tasks,
        repos_per_task,
    )
    return num_tasks


def finalize_collection(run_dir: Path) -> None:
    repo_file_shards = merge_jsonl_shards(run_dir, DATA_FILENAME)
    mined_repo_shards = merge_jsonl_shards(run_dir, MINNED_REPOS_FILENAME)
    logging.info(
        "Merged %d %s shards and %d %s shards",
        repo_file_shards,
        DATA_FILENAME,
        mined_repo_shards,
        MINNED_REPOS_FILENAME,
    )


def collect_dataset(
    run_dir: Path,
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

    all_repos = _get_repos(repo_min_stars, max_repos, run_dir)
    mining_end = dataset_directory_timestamp(run_dir)

    sorted_repos = _sort_repos_by_size(all_repos)
    if in_jobs_array:
        # Ensure a balanced distribution of repo sizes across tasks
        repos_partition = sorted_repos[task_id::num_tasks]
    else:
        repos_partition = sorted_repos

    _clean_previous_data(run_dir, data_filename, mined_filename)

    repos = _unmined_repos(repos_partition, run_dir, mined_filename)

    if not repos:
        logging.info("No repositories left to mine for this task")
        return

    write_lock = Lock()
    workers = max(1, min(DEFAULT_MINING_WORKERS, len(repos)))

    logging.info("Mining %d repositories with %d workers", len(repos), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _mine_and_persist_repo,
                repo,
                run_dir,
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
        "--prepare",
        action="store_true",
        help="Search GitHub once, cache the repo list, and print RUN_DIR and "
        "NUM_TASKS for the job array",
    )
    parser.add_argument(
        "--collect", action="store_true", help="Collect data from GitHub"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Merge the per-task output shards into single files",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a fresh timestamped run directory instead of using the latest",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Use this exact run directory instead of picking one "
        "automatically, so every array task shares one run",
    )

    array = parser.add_argument_group("job array (HPC partitioning)")
    array.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the job array. Mines only this task's share "
        "of the repos into its own sharded files",
    )
    array.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the job array. Splits the repos evenly "
        "across tasks",
    )
    array.add_argument(
        "--repos-per-task",
        type=int,
        default=10,
        help="Repos per array task; --prepare uses this to decide how many "
        "tasks to create",
    )

    limits = parser.add_argument_group("limits (for testing on smaller batches)")
    limits.add_argument(
        "--max-repos", type=int, default=1000, help="Limit number of repos processed"
    )
    limits.add_argument(
        "--repo-min-stars",
        type=int,
        default=0,
        help="Only include repos with at least this many stars",
    )
    limits.add_argument(
        "--repo-min-contributors",
        type=int,
        default=0,
        help="Drop repos with fewer than this many contributors",
    )

    args = parser.parse_args()

    no_stage_selected = not any((args.prepare, args.collect, args.finalize))
    if no_stage_selected:
        args.collect = True

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = resolve_dataset_directory(create_new_run=(args.new_run or args.prepare))
    logging.info("Using run directory: %s", run_dir)

    if args.prepare:
        num_tasks = prepare_collection(
            run_dir,
            max_repos=args.max_repos,
            repo_min_stars=args.repo_min_stars,
            repos_per_task=args.repos_per_task,
        )
        # submit.sh uses these prints to parse the RUN_DIR and NUM_TASKS
        print(f"RUN_DIR={run_dir}")
        print(f"NUM_TASKS={num_tasks}")
        return

    if args.collect:
        collect_dataset(
            run_dir,
            task_id=args.task_id,
            num_tasks=args.num_tasks,
            max_repos=args.max_repos,
            repo_min_stars=args.repo_min_stars,
        )
    if args.finalize:
        finalize_collection(run_dir)


if __name__ == "__main__":
    main()
