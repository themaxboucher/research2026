import os
import requests
from dotenv import load_dotenv
import base64

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

DEFAULT_TIMEOUT = 30

def _github_api_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

def search_repos(language: str = "Python", min_stars: int = 10_000, pushed_after: str = "2026-01-01") -> list[dict]:
    repos: list[dict] = []
    url_query = f"is:public+template:false+archived:false+language:{language}+stars:>={min_stars}+pushed:>{pushed_after}"
    url_sort = "sort=stars&order=desc"
    url_per_page = "per_page=100"
    url: str | None = f"https://api.github.com/search/repositories?q={url_query}&{url_sort}&{url_per_page}"
    while url:
        response = requests.get(url, headers=_github_api_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        repos.extend(data.get("items", []))
        url = response.links.get("next", {}).get("url")
    return repos

def get_repo_commits(repo_name: str, repo_owner: str, since: str = "2026-01-01") -> list[dict]:
    commits: list[dict] = []
    url_since = f"since={since}"
    url_per_page = "per_page=100"
    url: str | None = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?{url_since}&{url_per_page}"
    while url:
        response = requests.get(url, headers=_github_api_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        commits.extend(data)
        url = response.links.get("next", {}).get("url")
    return commits

def get_commit(repo_owner: str, repo_name: str, commit_sha: str) -> dict:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
    response = requests.get(url, headers=_github_api_headers(), timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()

def get_file_contents(repo_owner: str, repo_name: str, filepath: str, commit_sha: str) -> dict:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{filepath}?ref={commit_sha}"
    response = requests.get(url, headers=_github_api_headers(), timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    data["content"] = base64.b64decode(data["content"]).decode("utf-8") 
    return data