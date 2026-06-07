from github import (
    GITHUB_TOKENS,
    close_rate_limit_bars,
    get_commit,
    get_file_contents,
    get_repo_commits,
    get_repo_contributors,
    init_rate_limit_bars,
    search_repos,
)
from comments import get_comments_from_file, strip_comments_from_file
from storage import load_from_json, save_to_json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import tokenize

MAX_WORKERS = 8

# Knowledge cutoff dates
# Gemini 3.1 Pro (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro): January 2025
# GPT-5.5 (https://developers.openai.com/api/docs/models/gpt-5.5): December 1st, 2025
# Llama-3.1-8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# Qwen...

REPO_LANGUAGE = "Python"
CUTOFF_DATE = "2025-02-01"

CACHE_EVERY = 100


def get_files_from_commit(commit: dict) -> list[dict]:
    statuses_to_include = ["added", "modified", "removed"]
    return [
        file
        for file in commit["files"]
        if file.get("status") in statuses_to_include
        and file["filename"].endswith(".py")
    ]


def enrich_file(file: dict) -> dict:
    repo_owner = file["repo_owner"]
    repo_name = file["repo_name"]
    filename = file["filename"]
    status = file.get("status")
    sha = file.get("sha")
    parent_sha = file.get("parent_sha")

    is_removed = status == "removed"
    is_added = status == "added"

    content = None
    if not is_removed and sha:
        content = get_file_contents(repo_owner, repo_name, sha, filename)

    previous_content = None
    if not is_added and parent_sha:
        previous_content = get_file_contents(
            repo_owner, repo_name, parent_sha, filename
        )

    return {
        **file,
        "filepath": filename,
        "content": content,
        "previous_content": previous_content,
    }


def process_file(file: dict) -> dict | None:
    try:
        return {
            **file,
            "content_without_comments": strip_comments_from_file(file.get("content")),
            "previous_content_without_comments": strip_comments_from_file(
                file.get("previous_content")
            ),
            "comments": get_comments_from_file(file),
        }
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        logging.warning(
            "Skipping file %s/%s/%s: tokenization failed (%s)",
            file.get("repo_owner"),
            file.get("repo_name"),
            file.get("filename"),
            error,
        )
        return None


def add_contributor_count_to_repo(repo: dict) -> dict:
    if "contributors_count" in repo:
        return repo

    contributors = get_repo_contributors(repo["name"], repo["owner"]["login"])
    return {
        **repo,
        "contributors_count": len(contributors),
    }


def load_or_empty(directory: Path, filename: str) -> list[dict]:
    """Load a JSON file or return an empty list if the file does not exist."""
    try:
        return load_from_json(directory, filename)
    except FileNotFoundError:
        return []


def collect_dataset(
    run_dir: Path,
    max_repos: int | None = None,
    repo_min_stars: int = 0,
    repo_min_contributors: int = 0,
    max_commits_per_repo: int | None = None,
    max_commits: int | None = None,
    max_files: int | None = None,
) -> None:
    cache_dir = run_dir / "cache"

    repos: list[dict] = load_or_empty(cache_dir, "repos")
    commits: list[dict] = load_or_empty(cache_dir, "commits")
    detailed_commits: list[dict] = load_or_empty(cache_dir, "detailed_commits")
    file_contents: list[dict] = load_or_empty(cache_dir, "file_contents")

    if len(repos) == 0:
        logging.info("Searching for repositories...")
        repos = search_repos(
            language=REPO_LANGUAGE,
            min_stars=repo_min_stars,
            pushed_after=CUTOFF_DATE,
            limit=max_repos,
        )

    repos = repos[:max_repos]

    pipeline_position = len(GITHUB_TOKENS)

    with (
        ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor,
        logging_redirect_tqdm(),
    ):
        init_rate_limit_bars()
        try:
            repos_with_contributor_counts: list[dict] = list(
                tqdm(
                    executor.map(add_contributor_count_to_repo, repos),
                    total=len(repos),
                    desc="Getting repo contributors",
                    unit="repo",
                    position=pipeline_position,
                    leave=True,
                )
            )

            repos = [
                repo
                for repo in repos_with_contributor_counts
                if repo["contributors_count"] >= repo_min_contributors
            ]

            save_to_json(repos, cache_dir, "repos")

            logging.info("Total repositories: %d", len(repos))

            repos_with_commits_keys: set[tuple[str, str]] = {
                (commit["repo_owner"], commit["repo_name"]) for commit in commits
            }

            repos_to_fetch_commits: list[dict] = [
                repo
                for repo in repos
                if (repo["owner"]["login"], repo["name"]) not in repos_with_commits_keys
            ]

            for index, repo_commits in enumerate(
                tqdm(
                    executor.map(
                        lambda repo: (
                            repo,
                            get_repo_commits(
                                repo["name"],
                                repo["owner"]["login"],
                                since=CUTOFF_DATE,
                                limit=max_commits_per_repo,
                            ),
                        ),
                        repos_to_fetch_commits,
                    ),
                    total=len(repos_to_fetch_commits),
                    desc="Getting repo commits",
                    unit="repo",
                    position=pipeline_position + 1,
                    leave=True,
                )
            ):
                repo, commits_list = repo_commits
                commits_to_add: list[dict] = [
                    {
                        **commit,
                        "repo_name": repo["name"],
                        "repo_owner": repo["owner"]["login"],
                    }
                    for commit in commits_list
                ]
                commits.extend(commits_to_add)

                if index % CACHE_EVERY == 0:
                    save_to_json(commits_to_add, cache_dir, "commits")

            logging.info("Total commits: %d", len(commits))

            existing_detailed_commit_keys: set[tuple[str, str, str]] = {
                (commit["repo_owner"], commit["repo_name"], commit["sha"])
                for commit in detailed_commits
            }

            detailed_commits_to_fetch: list[dict] = [
                commit
                for commit in commits
                if (commit["repo_owner"], commit["repo_name"], commit["sha"])
                not in existing_detailed_commit_keys
            ][:max_commits]

            for index, detailed_commit in enumerate(
                tqdm(
                    executor.map(
                        lambda commit: get_commit(
                            commit["repo_owner"], commit["repo_name"], commit["sha"]
                        ),
                        detailed_commits_to_fetch,
                    ),
                    total=len(detailed_commits_to_fetch),
                    desc="Getting commit details",
                    unit="commit",
                    position=pipeline_position + 2,
                    leave=True,
                )
            ):
                detailed_commits.append(detailed_commit)
                if index % CACHE_EVERY == 0:
                    save_to_json(detailed_commits, cache_dir, "detailed_commits")

            commit_files: list[dict] = [
                {
                    **file,
                    "repo_owner": commit["repo_owner"],
                    "repo_name": commit["repo_name"],
                    "sha": commit["sha"],
                    "parent_sha": commit["parents"][0]["sha"] if commit["parents"] else None,
                    "date": commit["commit"]["committer"]["date"],
                }
                for commit in detailed_commits
                for file in get_files_from_commit(commit)
            ]

            logging.info("Total commit files: %d", len(commit_files))

            existing_file_keys: set[tuple[str, str, str, str]] = {
                (file["repo_owner"], file["repo_name"], file["filepath"], file["sha"])
                for file in file_contents
            }

            commit_files_to_fetch: list[dict] = [
                file
                for file in commit_files
                if (
                    file["repo_owner"],
                    file["repo_name"],
                    file["filename"],
                    file["sha"],
                )
                not in existing_file_keys
            ][:max_files]

            for index, file_content in enumerate(
                tqdm(
                    executor.map(
                        enrich_file,
                        commit_files_to_fetch,
                    ),
                    total=len(commit_files_to_fetch),
                    desc="Getting file contents",
                    unit="file",
                    position=pipeline_position + 3,
                    leave=True,
                )
            ):
                file_contents.append(file_content)
                if index % CACHE_EVERY == 0:
                    save_to_json(file_contents, cache_dir, "file_contents")

            processed_files: list[dict | None] = [
                process_file(file)
                for file in tqdm(
                    file_contents,
                    desc="Processing files",
                    unit="file",
                    position=pipeline_position + 4,
                    leave=True,
                )
            ]
            files: list[dict] = [file for file in processed_files if file is not None]

            skipped = len(processed_files) - len(files)
            logging.info(
                "Total files found: %d (skipped %d due to tokenization errors)",
                len(files),
                skipped,
            )
        finally:
            close_rate_limit_bars()

    if files:
        logging.info("Saving data to JSON...")
        save_to_json(files, run_dir, "files")
