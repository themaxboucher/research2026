import json
import logging
from datetime import datetime
from pathlib import Path

from storage.timestamped_dirs import (
    find_latest_timestamped_directory,
    new_timestamped_directory,
    resolve_timestamped_argument,
    timestamp_from_name,
)

DATASET_DIRECTORY_NAME = "datasets"
MANIFEST_FILENAME = "manifest"
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


def _latest_or_new_dataset_directory() -> Path:
    latest_directory = find_latest_dataset_directory()
    if latest_directory is not None:
        return latest_directory

    return create_new_dataset_directory()


def latest_dataset_directory() -> Path:
    latest_directory = find_latest_dataset_directory()
    if latest_directory is None:
        raise SystemExit("No dataset directory found.")
    return latest_directory


def resolve_dataset_directory(
    dataset_dir_arg: str | None,
    *,
    create_dataset: bool = False,
) -> Path:
    """Resolve the ``--dataset-dir`` CLI argument to a dataset directory, logging the choice.

    With an argument, resolve it to a path — created when `create_dataset` is
    set, so preparation can start a dataset at a chosen timestamp. With no
    argument and `create_dataset` set, start a new timestamped dataset
    (preparation); otherwise fall back to the latest existing dataset, creating
    one only if none exist yet.
    """
    if dataset_dir_arg:
        dataset_directory = dataset_directory_from_argument(dataset_dir_arg)
        if create_dataset:
            dataset_directory.mkdir(parents=True, exist_ok=True)
    elif create_dataset:
        dataset_directory = create_new_dataset_directory()
    else:
        dataset_directory = _latest_or_new_dataset_directory()
    logging.info("Using dataset directory: %s", dataset_directory)
    return dataset_directory


def write_manifest(dataset_dir: Path, manifest: dict) -> dict:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / (MANIFEST_FILENAME + ".json")).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
