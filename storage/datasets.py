from datetime import datetime
from pathlib import Path

from storage.timestamped_dirs import (
    find_latest_timestamped_directory,
    new_timestamped_directory,
    resolve_timestamped_argument,
    timestamp_from_name,
)

DATASET_DIRECTORY_NAME = "datasets"
# Repo-root datasets/ directory (storage/ sits directly under the repo root).
DATASET_DIRECTORY = Path(__file__).parent.parent / DATASET_DIRECTORY_NAME


def dataset_directory_from_argument(dataset_dir_argument: str) -> Path:
    return resolve_timestamped_argument(
        dataset_dir_argument, DATASET_DIRECTORY, label="dataset directory"
    )


def create_new_dataset_directory() -> Path:
    return new_timestamped_directory(DATASET_DIRECTORY)


def dataset_directory_timestamp(dataset_directory: Path) -> datetime:
    return timestamp_from_name(dataset_directory.name)


def find_latest_dataset_directory() -> Path | None:
    return find_latest_timestamped_directory(DATASET_DIRECTORY)


def resolve_dataset_directory() -> Path:
    latest_directory = find_latest_dataset_directory()
    if latest_directory is not None:
        return latest_directory

    return create_new_dataset_directory()


def latest_dataset_directory() -> Path:
    latest_directory = find_latest_dataset_directory()
    if latest_directory is None:
        raise SystemExit("No dataset directory found.")
    return latest_directory
