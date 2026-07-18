from datetime import datetime
from pathlib import Path
from collect.constants import DATASET_DIRECTORY_NAME

DATASET_DIRECTORY = Path(__file__).parent / DATASET_DIRECTORY_NAME
DATASET_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


def create_new_dataset_directory() -> Path:
    new_dataset_directory = DATASET_DIRECTORY / datetime.now().strftime(DATASET_TIMESTAMP_FORMAT)
    new_dataset_directory.mkdir(parents=True, exist_ok=True)
    return new_dataset_directory


def dataset_directory_timestamp(dataset_directory: Path) -> datetime:
    return datetime.strptime(dataset_directory.name, DATASET_TIMESTAMP_FORMAT)


def find_latest_dataset_directory() -> Path | None:
    if not DATASET_DIRECTORY.exists():
        return None

    existing_dataset_directories = sorted(
        path for path in DATASET_DIRECTORY.iterdir() if path.is_dir()
    )
    if not existing_dataset_directories:
        return None

    return existing_dataset_directories[-1]


def resolve_dataset_directory() -> Path:
    latest_dataset_directory = find_latest_dataset_directory()
    if latest_dataset_directory is not None:
        return latest_dataset_directory

    return create_new_dataset_directory()


def require_latest_dataset_directory() -> Path:
    latest_dataset_directory = find_latest_dataset_directory()
    if latest_dataset_directory is None:
        raise SystemExit(
            "No dataset directory found."
        )
    return latest_dataset_directory
