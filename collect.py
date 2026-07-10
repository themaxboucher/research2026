from github import search_repos
from comments import get_comments_from_file, strip_comments_from_file
from runs import run_directory_timestamp
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
import tokenize
from pydriller import Repository
from datetime import datetime

# Knowledge cutoff dates
# Gemini 3.1 Pro (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro): January 2025
# GPT-5.5 (https://developers.openai.com/api/docs/models/gpt-5.5): December 1st, 2025
# Llama-3.1-8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# Qwen...

REPO_LANGUAGE = "Python"
CUTOFF_DATE = "2025-12-02"
DEFAULT_MINING_WORKERS = 16

DATA_FILENAME = "repo_files"
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


def _collect_repo_files(repo: Repository, repo_full_name: str) -> list[dict]:
    repo_files = []

    for commit in repo.traverse_commits():
        for file in commit.modified_files:
            is_python_file = file.filename.endswith(".py")
            is_rename_change = file.change_type.name == "RENAME"

            if not is_python_file:
                continue
            if is_rename_change:
                continue

            error = None

            try:
                source_code_without_comments = strip_comments_from_file(
                    file.source_code
                )
                previous_source_code_without_comments = strip_comments_from_file(
                    file.source_code_before
                )
                comments = get_comments_from_file(
                    file.source_code, file.source_code_before
                )
            except (tokenize.TokenError, IndentationError, SyntaxError) as e:
                logging.warning(
                    "Could not process file %s: tokenization failed: %s",
                    file.filename,
                    e,
                )
                source_code_without_comments = None
                previous_source_code_without_comments = None
                comments = None
                error = str(e)

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
                    "source_code_without_comments": source_code_without_comments,
                    "previous_source_code_without_comments": previous_source_code_without_comments,
                    "comments": comments,
                    "nloc": file.nloc,
                    "complexity": file.complexity,
                    "token_count": file.token_count,
                    "error": error,
                }
            )

    return repo_files


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

    return _collect_repo_files(repo, repo_full_name)


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
    mining_end = run_directory_timestamp(run_dir)

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
