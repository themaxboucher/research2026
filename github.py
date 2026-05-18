import os
import requests
from dotenv import load_dotenv
import base64
import logging

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
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logging.error(f"Error searching repos: {e}")
            return repos
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
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logging.error(f"Error getting repo commits: {e}")
            return commits
        data = response.json()
        commits.extend(data)
        url = response.links.get("next", {}).get("url")
    return commits

def get_commit(repo_owner: str, repo_name: str, commit_sha: str) -> dict:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
    response = requests.get(url, headers=_github_api_headers(), timeout=DEFAULT_TIMEOUT)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Error getting commit: {e}")
        return None
    return response.json()

def get_file_contents(repo_owner: str, repo_name: str, filepath: str, commit_sha: str) -> dict:
    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{commit_sha}/{filepath}"
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Error getting file contents: {url} {e}")
        return None
    data = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "filepath": filepath,
        "sha": commit_sha,
        "content": response.text,
    }
    return data