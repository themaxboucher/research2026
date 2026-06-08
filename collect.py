from github import search_repos
from comments import get_comments_from_file, strip_comments_from_file
from storage import append_to_jsonl
from tqdm.auto import tqdm
import logging
from pathlib import Path
import tokenize
from pydriller import Repository
from datetime import datetime

# Knowledge cutoff dates
# Gemini 3.1 Pro (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro): January 2025
# GPT-5.5 (https://developers.openai.com/api/docs/models/gpt-5.5): December 1st, 2025
# Llama-3.1-8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# Qwen...

REPO_LANGUAGE = "Python"
CUTOFF_DATE = "2025-02-01"


def mine_repo(repo_url: str, branch: str, since: str) -> list[dict]:
    datetime_since = datetime.strptime(since, "%Y-%m-%d")

    repo = Repository(
        repo_url,
        since=datetime_since,
        only_in_branch=branch,
        only_no_merge=True,
        only_modifications_with_file_types=[".py"],
    )

    repo_files = []

    for commit in repo.traverse_commits():
        for file in commit.modified_files:
            is_python_file = file.filename.endswith(".py")
            is_rename_change = file.change_type.name == "RENAME"

            if not is_python_file:
                continue
            if is_rename_change:
                continue

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
            except (tokenize.TokenError, IndentationError, SyntaxError) as error:
                logging.warning(
                    "Skipping file %s: tokenization failed: %s",
                    file.filename,
                    error,
                )
                continue

            repo_files.append(
                {
                    "filename": file.filename,
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
                }
            )

    return repo_files


def collect_dataset(
    run_dir: Path,
    max_repos: int | None = None,
    repo_min_stars: int = 0,
) -> None:

    logging.info("Searching for repositories...")
    repos = search_repos(
        language=REPO_LANGUAGE,
        min_stars=repo_min_stars,
        pushed_after=CUTOFF_DATE,
        limit=max_repos,
    )

    repos = repos[:max_repos]

    for repo in tqdm(repos, desc="Mining repositories", unit="repo", leave=True):
        repo_files = mine_repo(repo["html_url"], repo["default_branch"], CUTOFF_DATE)
        append_to_jsonl(repo_files, run_dir, "repo_files")
