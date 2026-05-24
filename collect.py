from github import search_repos, get_repo_commits, get_commit, get_file_contents
from comments import get_comments_from_file, strip_comments_from_file
from storage import save_to_json
from concurrent.futures import ThreadPoolExecutor
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

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        logging.info("Getting commits for repositories...")
        commits_by_repo: list[tuple[dict, list[dict]]] = list(executor.map(
            lambda repo: (repo, get_repo_commits(repo["name"], repo["owner"]["login"], since=CUTOFF_DATE)),
            repos,
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

        logging.info("Getting detailed commits for repositories...")
        detailed_commits: list[dict] = list(executor.map(
            lambda commit: get_commit(commit["repo_owner"], commit["repo_name"], commit["sha"]),
            commits,
        ))

        logging.info("Total detailed commits fetched: %d", len(detailed_commits))

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

        logging.info("Getting file contents for commits...")
        file_contents: list[dict] = list(executor.map(
            lambda file: get_file_contents(file["repo_owner"], file["repo_name"], file["filename"], file["sha"]),
            commit_files,
        ))

        processed_files: list[dict | None] = [process_file(file) for file in file_contents]
        files: list[dict] = [file for file in processed_files if file is not None]

        skipped = len(processed_files) - len(files)
        logging.info("Total files found: %d (skipped %d due to tokenization errors)", len(files), skipped)

    if files:
        logging.info("Saving data to JSON...")
        save_to_json(files, "files")