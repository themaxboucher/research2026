from github import (
    GITHUB_TOKENS,
    close_rate_limit_bars,
    get_commit,
    get_file_contents,
    get_repo_commits,
    init_rate_limit_bars,
    search_repos,
)
from comments import get_comments_from_file, strip_comments_from_file
from storage import save_to_json
from concurrent.futures import ThreadPoolExecutor
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
REPO_TOPIC = "python"
REPO_MIN_STARS = 10_000
CUTOFF_DATE = "2025-02-01"

def get_files_from_commit(commit: dict) -> list[dict]:
    return [
        file 
        for file in commit["files"] 
        if file.get("status") == "added" and 
        file["filename"].endswith(".py")
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
            file.get("repo_owner"), file.get("repo_name"), file.get("filename"), e,
        )
        return None

def collect_dataset() -> None:
    logging.info("Searching for repositories...")

    repos = search_repos(language=REPO_LANGUAGE, topic=REPO_TOPIC, min_stars=REPO_MIN_STARS, pushed_after=CUTOFF_DATE)

    logging.info("Total repositories found: %d", len(repos))

    pipeline_position = len(GITHUB_TOKENS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, logging_redirect_tqdm():
        init_rate_limit_bars()
        try:
            commits_by_repo: list[tuple[dict, list[dict]]] = list(tqdm(
                executor.map(
                    lambda repo: (repo, get_repo_commits(repo["name"], repo["owner"]["login"], since=CUTOFF_DATE)),
                    repos,
                ),
                total=len(repos),
                desc="Getting repo commits",
                unit="repo",
                position=pipeline_position,
                leave=True,
            ))

            commits: list[dict] = [
                {
                    **commit,
                    "repo_name": repo["name"],
                    "repo_owner": repo["owner"]["login"]
                }
                for repo, commits_list in commits_by_repo
                for commit in commits_list
            ]

            logging.info("Total commits found: %d", len(commits))

            detailed_commits: list[dict] = list(tqdm(
                executor.map(
                    lambda commit: get_commit(commit["repo_owner"], commit["repo_name"], commit["sha"]),
                    commits,
                ),
                total=len(commits),
                desc="Getting commit details",
                unit="commit",
                position=pipeline_position + 1,
                leave=True,
            ))

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

            logging.info("Total commit files found: %d", len(commit_files))

            file_contents: list[dict] = list(tqdm(
                executor.map(
                    lambda file: get_file_contents(file["repo_owner"], file["repo_name"], file["filename"], file["sha"]),
                    commit_files,
                ),
                total=len(commit_files),
                desc="Getting file contents",
                unit="file",
                position=pipeline_position + 2,
                leave=True,
            ))

            processed_files: list[dict | None] = [
                process_file(file)
                for file in tqdm(
                    file_contents,
                    desc="Processing files",
                    unit="file",
                    position=pipeline_position + 3,
                    leave=True,
                )
            ]
            files: list[dict] = [file for file in processed_files if file is not None]

            skipped = len(processed_files) - len(files)
            logging.info("Total files found: %d (skipped %d due to tokenization errors)", len(files), skipped)
        finally:
            close_rate_limit_bars()

    if files:
        logging.info("Saving data to JSON...")
        save_to_json(files, "files")