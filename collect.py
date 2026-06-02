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
REPO_MIN_STARS = 20
REPO_MIN_CONTRIBUTORS = 5
CUTOFF_DATE = "2025-02-01"

SAVE_EVERY = 100


def get_files_from_commit(commit: dict) -> list[dict]:
    return [
        file
        for file in commit["files"]
        if file.get("status") == "added" and file["filename"].endswith(".py")
    ]


def process_file(file: dict) -> dict | None:
    try:
        return {
            **file,
            "content_without_comments": strip_comments_from_file(file["content"]),
            "comments": get_comments_from_file(file["content"]),
        }
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        logging.warning(
            "Skipping file %s/%s/%s: tokenization failed (%s)",
            file.get("repo_owner"),
            file.get("repo_name"),
            file.get("filepath"),
            e,
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
            min_stars=REPO_MIN_STARS, 
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
                if repo["contributors_count"] >= REPO_MIN_CONTRIBUTORS
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

                if index % SAVE_EVERY == 0:
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
                if index % SAVE_EVERY == 0:
                    save_to_json(detailed_commits, cache_dir, "detailed_commits")

            commit_files: list[dict] = [
                {
                    **file,
                    "repo_owner": commit["repo_owner"],
                    "repo_name": commit["repo_name"],
                    "sha": commit["sha"],
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
                        lambda file: get_file_contents(
                            file["repo_owner"],
                            file["repo_name"],
                            file["filename"],
                            file["sha"],
                        ),
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
                if index % SAVE_EVERY == 0:
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
