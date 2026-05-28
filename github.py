import itertools
import logging
import os
import threading
import time

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
RATE_LIMIT_DEFAULT_TOTAL = 5000

def _load_github_tokens() -> list[str]:
    raw = os.getenv("GITHUB_TOKENS") or os.getenv("GITHUB_TOKEN") or ""
    return [token.strip() for token in raw.split(",") if token.strip()]

GITHUB_TOKENS = _load_github_tokens()
_token_cycle = itertools.cycle(GITHUB_TOKENS) if GITHUB_TOKENS else itertools.cycle([None])
_token_lock = threading.Lock()

_rate_limit_bars: list[tqdm] = []
_rate_limit_lock = threading.Lock()

def _format_reset_time(reset_timestamp: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(reset_timestamp))

def init_rate_limit_bars() -> None:
    global _rate_limit_bars
    if _rate_limit_bars or not GITHUB_TOKENS:
        return
    _rate_limit_bars = [
        tqdm(
            total=RATE_LIMIT_DEFAULT_TOTAL,
            desc=f"Token {i + 1} rate limit",
            position=i,
            leave=True,
            bar_format="{desc}: {n}/{total} used |{bar}|{postfix}",
        )
        for i in range(len(GITHUB_TOKENS))
    ]

def close_rate_limit_bars() -> None:
    global _rate_limit_bars
    for bar in _rate_limit_bars:
        bar.close()
    _rate_limit_bars = []

def _update_rate_limit_bar(token: str | None, response: requests.Response) -> None:
    if token is None or not _rate_limit_bars:
        return
    remaining_raw = response.headers.get("X-RateLimit-Remaining")
    if remaining_raw is None:
        return
    try:
        idx = GITHUB_TOKENS.index(token)
    except ValueError:
        return
    if idx >= len(_rate_limit_bars):
        return
    try:
        remaining = int(remaining_raw)
        limit_raw = response.headers.get("X-RateLimit-Limit")
        limit = int(limit_raw) if limit_raw else _rate_limit_bars[idx].total
    except ValueError:
        return
    reset_raw = response.headers.get("X-RateLimit-Reset")
    with _rate_limit_lock:
        bar = _rate_limit_bars[idx]
        bar.total = limit
        bar.n = max(0, limit - remaining)
        if reset_raw:
            try:
                bar.set_postfix_str(
                    f"resets at {_format_reset_time(int(reset_raw))}",
                    refresh=False,
                )
            except ValueError:
                pass
        bar.refresh()

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

def _github_get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    last_response: requests.Response | None = None

    for attempt in range(MAX_RETRIES):
        try:
            for token in _tokens_to_try():
                response = requests.get(url, headers=_github_api_headers(token), **kwargs)
                last_response = response
                if not _is_rate_limited(response):
                    response.raise_for_status()
                    _update_rate_limit_bar(token, response)
                    return response
                _update_rate_limit_bar(token, response)

            if last_response is None:
                raise RuntimeError(
                    f"GitHub API request returned no response after trying all tokens: {url}"
                )

            reset_raw = last_response.headers.get("X-RateLimit-Reset")
            if reset_raw:
                wait_buffer_seconds = 1
                wait_seconds = max(0, int(reset_raw) - int(time.time())) + wait_buffer_seconds
                time.sleep(wait_seconds)
                return _github_get(url, **kwargs)

            status = last_response.status_code
            raise RuntimeError(
                f"GitHub API rate limit exhausted for all tokens (HTTP {status}); "
                f"no X-RateLimit-Reset header to wait on: {url}"
            )
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt + 1 >= MAX_RETRIES:
                raise RuntimeError(
                    f"GitHub API GET failed after {MAX_RETRIES} retries: {url}"
                ) from e
            wait = 2**attempt
            logging.warning(
                "GitHub API error (attempt %s/%s), retrying in %ss: %s",
                attempt + 1,
                MAX_RETRIES,
                wait,
                e,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"GitHub API GET failed after {MAX_RETRIES} connection retries: {url}"
    )

def search_repos(language: str, topic: str, min_stars: int, pushed_after: str) -> list[dict]:
    repos: list[dict] = []
    url_query = f"is:public+template:false+archived:false+language:{language}+topic:{topic}+stars:>={min_stars}+pushed:>{pushed_after}"
    url_sort = "sort=stars&order=desc"
    url_per_page = "per_page=100"
    url: str | None = f"https://api.github.com/search/repositories?q={url_query}&{url_sort}&{url_per_page}"
    while url:
        response = _github_get(url)
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
        data = response.json()
        commits.extend(data)
        url = response.links.get("next", {}).get("url")
    return commits

def get_commit(repo_owner: str, repo_name: str, commit_sha: str) -> dict:
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
    response = _github_get(url)
    data = response.json()
    data["repo_owner"] = repo_owner
    data["repo_name"] = repo_name
    return data

def get_file_contents(repo_owner: str, repo_name: str, filepath: str, commit_sha: str) -> dict:
    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{commit_sha}/{filepath}"
    response = _github_get(url)
    data = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "filepath": filepath,
        "sha": commit_sha,
        "content": response.text,
    }
    return data
