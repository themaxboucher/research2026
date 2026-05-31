import itertools
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
GITHUB_RATE_LIMIT_PER_HOUR = 5000  # GitHub REST API has a rate limit of 5000 requests per hour for authenticated users (https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api?apiVersion=2026-03-10)
GITHUB_SEARCH_RATE_LIMIT_PER_MINUTE = 30  # GitHub REST API limit search queries to 30 requests per minute for authenticated users (https://docs.github.com/en/rest/search/search?apiVersion=2026-03-10#about-search)
GITHUB_SEARCH_RESULT_LIMIT = 1000  # GitHub REST API limit search queries to 1000 results (https://docs.github.com/en/rest/search/search?apiVersion=2026-03-10#search-repositories)
SEARCH_MAX_WORKERS = 8

SearchPartition = tuple[int, int, str, str]


def _load_github_tokens() -> list[str]:
    raw = os.getenv("GITHUB_TOKENS") or os.getenv("GITHUB_TOKEN") or ""
    return [token.strip() for token in raw.split(",") if token.strip()]


GITHUB_TOKENS = _load_github_tokens()
_token_cycle = (
    itertools.cycle(GITHUB_TOKENS) if GITHUB_TOKENS else itertools.cycle([None])
)
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
            total=GITHUB_RATE_LIMIT_PER_HOUR,
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
                response = requests.get(
                    url, headers=_github_api_headers(token), **kwargs
                )
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
                wait_seconds = (
                    max(0, int(reset_raw) - int(time.time())) + wait_buffer_seconds
                )
                time.sleep(wait_seconds)
                return _github_get(url, **kwargs)

            status = last_response.status_code
            raise RuntimeError(
                f"GitHub API rate limit exhausted for all tokens (HTTP {status}); "
                f"no X-RateLimit-Reset header to wait on: {url}"
            )
        except (
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
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


def _build_search_query(
    language: str,
    min_stars: int,
    max_stars: int | None,
    min_pushed_date: str,
    max_pushed_date: str | None = None,
) -> str:
    stars_qualifier = (
        f"stars:{min_stars}..{max_stars}"
        if max_stars is not None
        else f"stars:>={min_stars}"
    )
    pushed_qualifier = (
        f"pushed:{min_pushed_date}..{max_pushed_date}"
        if max_pushed_date is not None
        else f"pushed:>{min_pushed_date}"
    )
    return f"is:public+template:false+archived:false+language:{language}+{stars_qualifier}+{pushed_qualifier}"


def _count_repos_in_partition(language: str, partition: SearchPartition) -> int:
    min_stars, max_stars, min_pushed_date, max_pushed_date = partition
    query = _build_search_query(
        language, min_stars, max_stars, min_pushed_date, max_pushed_date
    )
    url = f"https://api.github.com/search/repositories?q={query}&per_page=1"
    response = _github_get(url)
    return response.json().get("total_count", 0)


def _get_highest_star_count(language: str, min_stars: int, pushed_after: str) -> int:
    query = _build_search_query(language, min_stars, None, pushed_after)
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=1"
    response = _github_get(url)
    items = response.json().get("items", [])
    if not items:
        return min_stars
    return items[0]["stargazers_count"]


def _parse_search_date(search_date: str) -> date:
    return datetime.strptime(search_date, "%Y-%m-%d").date()


def _format_search_date(search_date: date) -> str:
    return search_date.isoformat()


def _get_first_partition_pushed_date(pushed_after: str) -> str:
    first_partition_pushed_date = _parse_search_date(pushed_after) + timedelta(days=1)
    return _format_search_date(first_partition_pushed_date)


def _split_star_partition(partition: SearchPartition) -> list[SearchPartition]:
    min_stars, max_stars, min_pushed_date, max_pushed_date = partition
    midpoint_stars = (min_stars + max_stars) // 2
    return [
        (min_stars, midpoint_stars, min_pushed_date, max_pushed_date),
        (midpoint_stars + 1, max_stars, min_pushed_date, max_pushed_date),
    ]


def _split_pushed_partition(partition: SearchPartition) -> list[SearchPartition]:
    min_stars, max_stars, min_pushed_date, max_pushed_date = partition
    min_pushed = _parse_search_date(min_pushed_date)
    max_pushed = _parse_search_date(max_pushed_date)
    midpoint_pushed = min_pushed + (max_pushed - min_pushed) // 2
    next_pushed = midpoint_pushed + timedelta(days=1)
    return [
        (
            min_stars,
            max_stars,
            _format_search_date(min_pushed),
            _format_search_date(midpoint_pushed),
        ),
        (
            min_stars,
            max_stars,
            _format_search_date(next_pushed),
            _format_search_date(max_pushed),
        ),
    ]


def _partition_repo_searches(
    language: str, min_stars: int, pushed_after: str
) -> list[SearchPartition]:
    highest_star_count = _get_highest_star_count(language, min_stars, pushed_after)
    min_pushed_date = _get_first_partition_pushed_date(pushed_after)
    max_pushed_date = _format_search_date(date.today())
    safe_partitions: list[SearchPartition] = []
    partitions_to_check: list[SearchPartition] = [
        (min_stars, highest_star_count, min_pushed_date, max_pushed_date)
    ]
    while partitions_to_check:
        partition = partitions_to_check.pop()
        (
            min_partition_stars,
            max_partition_stars,
            min_partition_pushed_date,
            max_partition_pushed_date,
        ) = partition
        count = _count_repos_in_partition(language, partition)
        if count <= GITHUB_SEARCH_RESULT_LIMIT:
            safe_partitions.append(partition)
            continue
        if min_partition_stars < max_partition_stars:
            partitions_to_check.extend(_split_star_partition(partition))
            continue
        if _parse_search_date(min_partition_pushed_date) < _parse_search_date(
            max_partition_pushed_date
        ):
            partitions_to_check.extend(_split_pushed_partition(partition))
            continue
        if count > GITHUB_SEARCH_RESULT_LIMIT:
            logging.warning(
                "Star count %d and pushed date %s has %d repos, exceeding the search limit of %d; some repos will be missed",
                min_partition_stars,
                min_partition_pushed_date,
                count,
                GITHUB_SEARCH_RESULT_LIMIT,
            )
        safe_partitions.append(partition)
    return safe_partitions


def _search_repos_in_partition(language: str, partition: SearchPartition) -> list[dict]:
    repos: list[dict] = []
    min_stars, max_stars, min_pushed_date, max_pushed_date = partition
    query = _build_search_query(
        language, min_stars, max_stars, min_pushed_date, max_pushed_date
    )
    url: str | None = (
        f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=100"
    )
    while url:
        response = _github_get(url)
        repos.extend(response.json().get("items", []))
        url = response.links.get("next", {}).get("url")
    return repos


def _deduplicate_repos(repos: list[dict]) -> list[dict]:
    repos_by_id = {repo["id"]: repo for repo in repos}
    return list(repos_by_id.values())


def search_repos(language: str, min_stars: int, pushed_after: str) -> list[dict]:
    search_partitions = _partition_repo_searches(language, min_stars, pushed_after)
    repos: list[dict] = []

    with ThreadPoolExecutor(max_workers=SEARCH_MAX_WORKERS) as executor:
        repo_lists = list(
            tqdm(
                executor.map(
                    lambda partition: _search_repos_in_partition(language, partition),
                    search_partitions,
                ),
                total=len(search_partitions),
                desc="Searching repos",
                unit="partition",
            )
        )

    for repo_list in repo_lists:
        repos.extend(repo_list)
    return _deduplicate_repos(repos)


def get_repo_contributors(repo_name: str, repo_owner: str) -> list[dict]:
    contributors: list[dict] = []
    url_per_page = "per_page=100"
    url: str | None = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors?{url_per_page}"
    )
    while url:
        response = _github_get(url)
        data = response.json() if response.content else []
        contributors.extend(data)
        url = response.links.get("next", {}).get("url")
    return contributors


def get_repo_commits(
    repo_name: str, repo_owner: str, since: str = "2026-01-01"
) -> list[dict]:
    commits: list[dict] = []
    url_since = f"since={since}"
    url_per_page = "per_page=100"
    url: str | None = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?{url_since}&{url_per_page}"
    )
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


def get_file_contents(
    repo_owner: str, repo_name: str, filepath: str, commit_sha: str
) -> dict:
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
