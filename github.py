import itertools
import logging
import os
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TIMEOUT = 30

def _load_github_tokens() -> list[str]:
    raw = os.getenv("GITHUB_TOKENS") or os.getenv("GITHUB_TOKEN") or ""
    return [token.strip() for token in raw.split(",") if token.strip()]

GITHUB_TOKENS = _load_github_tokens()
_token_cycle = itertools.cycle(GITHUB_TOKENS) if GITHUB_TOKENS else itertools.cycle([None])
_token_lock = threading.Lock()

def _next_token() -> str | None:
    with _token_lock:
        return next(_token_cycle)

def _github_api_headers(token: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _is_rate_limited(response: requests.Response) -> bool:
    if response.status_code not in (403, 429):
        return False
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        return int(remaining) == 0
    return True

def _tokens_to_try() -> list[str | None]:
    if not GITHUB_TOKENS:
        return [None]
    first = _next_token()
    return [first] + [token for token in GITHUB_TOKENS if token != first]

def _log_successful_request(response: requests.Response) -> None:
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        logging.info(
            "GitHub API GET %s -> %s (remaining=%s)",
            response.url,
            response.status_code,
            remaining,
        )
    else:
        logging.info(
            "GitHub API GET %s -> %s",
            response.url,
            response.status_code,
        )

def _github_get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    last_response: requests.Response | None = None

    for token in _tokens_to_try():
        response = requests.get(url, headers=_github_api_headers(token), **kwargs)
        last_response = response
        if not _is_rate_limited(response):
            if response.ok:
                _log_successful_request(response)
            return response
        logging.warning("GitHub rate limit hit, trying next token")

    if last_response is None:
        raise RuntimeError("No GitHub API response")

    reset_raw = last_response.headers.get("X-RateLimit-Reset")
    if reset_raw:
        wait_seconds = max(0, int(reset_raw) - int(time.time())) + 1
        logging.info("All tokens rate limited, sleeping %s seconds", wait_seconds)
        time.sleep(wait_seconds)
        return _github_get(url, **kwargs)

    return last_response

def search_repos(language: str, topic: str, min_stars: int, pushed_after: str) -> list[dict]:
    repos: list[dict] = []
    url_query = f"is:public+template:false+archived:false+language:{language}+topic:{topic}+stars:>={min_stars}+pushed:>{pushed_after}"
    url_sort = "sort=stars&order=desc"
    url_per_page = "per_page=100"
    url: str | None = f"https://api.github.com/search/repositories?q={url_query}&{url_sort}&{url_per_page}"
    while url:
        response = _github_get(url)
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
        response = _github_get(url)
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
    response = _github_get(url)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Error getting commit: {e}")
        return None
    data = response.json()
    data["repo_owner"] = repo_owner
    data["repo_name"] = repo_name
    return data

def get_file_contents(repo_owner: str, repo_name: str, filepath: str, commit_sha: str) -> dict:
    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{commit_sha}/{filepath}"
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"Error getting file contents: {url} {e}")
        return None
    _log_successful_request(response)
    data = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "filepath": filepath,
        "sha": commit_sha,
        "content": response.text,
    }
    return data
