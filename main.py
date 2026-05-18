from github import search_repos, get_repo_commits, get_commit, get_file_contents
from comments import get_comments_from_patch, strip_comments_from_file
from storage import save_to_json
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)

MAX_WORKERS = 8
CUTOFF_DATE = "2026-01-01"

def get_files_from_commit(commit: dict) -> list[dict]:
    return [
        file 
        for file in commit["files"] 
        if file.get("status") != "removed" and 
        file["filename"].endswith(".py") and 
        file.get("patch")
    ]

def main():
    logging.info("Searching for repositories...")

    repos = search_repos(pushed_after=CUTOFF_DATE)

    logging.info("Total repositories found: %d", len(repos))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        logging.info("Getting commits for repositories...")
        commits_by_repo: list[tuple[dict, list[dict]]] = list(executor.map(
            lambda repo: (repo, get_repo_commits(repo["name"], repo["owner"]["login"], since=CUTOFF_DATE)),
            repos[:20], # only get the first 20 repos to save time
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
            commits[:100], # only get the first 100 commits to save time
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

        logging.info("Getting comments from commits...")   

        comments: list[dict] = [
            {
                **comment,
                "repo_owner": file["repo_owner"],
                "repo_name": file["repo_name"],
                "sha": file["sha"],
            }
            for file in commit_files
            for comment in get_comments_from_patch(file["filename"], file["patch"])
        ]

        logging.info("Total comments found: %d", len(comments))

        logging.info("Getting file contents for commits...")
        file_contents: list[dict] = list(executor.map(
            lambda file: get_file_contents(file["repo_owner"], file["repo_name"], file["filename"], file["sha"]),
            commit_files,
        ))

        files: list[dict] = [
            {
                **file,
                "content_without_comments": strip_comments_from_file(file["content"]),
            }
            for file in file_contents
        ]
    
        logging.info("Total files found: %d", len(files))

    logging.info("Saving data to JSON...")
    save_to_json(comments, "comments")
    save_to_json(files, "files")

if __name__ == "__main__":
    main()
