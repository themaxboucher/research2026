import argparse
import logging
import math
from pathlib import Path

from collect.constants import (
    DEFAULT_MAX_REPOS,
    DEFAULT_REPOS_PER_TASK,
)
from collect.repos import get_repos
from storage.datasets import resolve_dataset_directory


def _prepare(
    dataset_directory: Path,
    max_repos: int | None,
    repo_min_stars: int,
    repos_per_task: int,
) -> int:
    repos = get_repos(repo_min_stars, max_repos, dataset_directory)
    num_tasks = max(1, math.ceil(len(repos) / repos_per_task))
    logging.info(
        "Prepared %d repositories into %d tasks (<=%d repos each)",
        len(repos),
        num_tasks,
        repos_per_task,
    )
    return num_tasks


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Existing dataset directory to resume instead of creating a new one: "
        "a timestamp name (resolved under datasets/) or a path ending in a timestamp",
    )
    parser.add_argument(
        "--repos-per-task",
        type=int,
        default=DEFAULT_REPOS_PER_TASK,
        help="Repos per array task",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=DEFAULT_MAX_REPOS,
        help="Limit number of repos processed",
    )
    parser.add_argument(
        "--repo-min-stars",
        type=int,
        default=0,
        help="Only include repos with at least this many stars",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = resolve_dataset_directory(args.dataset_dir, create_dataset=True)

    num_tasks = _prepare(
        dataset_directory,
        max_repos=args.max_repos,
        repo_min_stars=args.repo_min_stars,
        repos_per_task=args.repos_per_task,
    )
    # submit.sh uses these prints to parse the DATASET_DIR and NUM_TASKS
    print(f"DATASET_DIR={dataset_directory}")
    print(f"NUM_TASKS={num_tasks}")


if __name__ == "__main__":
    main()
