import logging
from pathlib import Path

from collect.constants import CUTOFF_DATE, REPO_LANGUAGE, REPOS_CACHE_FILENAME
from collect.github import search_repos
from storage import append_to_jsonl, load_from_jsonl


def get_repos(
    repo_min_stars: int, max_repos: int | None, dataset_directory: Path
) -> list[dict]:
    repos_cache_path = dataset_directory / f"{REPOS_CACHE_FILENAME}.jsonl"
    if repos_cache_path.exists():
        logging.info("Loading cached repositories from %s", repos_cache_path)
        return load_from_jsonl(dataset_directory, REPOS_CACHE_FILENAME)

    logging.info("Searching for repositories with at least %d stars", repo_min_stars)
    repos = search_repos(
        language=REPO_LANGUAGE,
        min_stars=repo_min_stars,
        pushed_after=CUTOFF_DATE,
        limit=max_repos,
    )[:max_repos]

    append_to_jsonl(repos, dataset_directory, REPOS_CACHE_FILENAME)

    return repos
