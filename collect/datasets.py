from datetime import datetime
from pathlib import Path
from collect.constants import DATASET_DIRECTORY

TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


def is_dataset_timestamp_format(directory_name: str) -> bool:
    try:
        datetime.strptime(directory_name, TIMESTAMP_FORMAT)
        return True
    except ValueError:
        return False


def dataset_directory_from_argument(dataset_dir_argument: str) -> Path:
    dataset_directory = Path(dataset_dir_argument)
    if not is_dataset_timestamp_format(dataset_directory.name):
        raise SystemExit(
            f"Invalid dataset directory '{dataset_dir_argument}': the directory "
            "name must be a timestamp in the format YYYY-MM-DDTHH-MM-SS"
        )
    if dataset_directory.parent == Path("."):
        return DATASET_DIRECTORY / dataset_directory.name
    return dataset_directory


def create_new_dataset_directory() -> Path:
    new_dataset_directory = DATASET_DIRECTORY / datetime.now().strftime(TIMESTAMP_FORMAT)
    new_dataset_directory.mkdir(parents=True, exist_ok=True)
    return new_dataset_directory


def dataset_directory_timestamp(dataset_directory: Path) -> datetime:
    return datetime.strptime(dataset_directory.name, TIMESTAMP_FORMAT)


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
    latest_directory = find_latest_dataset_directory()
    if latest_directory is not None:
        return latest_directory

    return create_new_dataset_directory()


def latest_dataset_directory() -> Path:
    latest_directory = find_latest_dataset_directory()
    if latest_directory is None:
        raise SystemExit(
            "No dataset directory found."
        )
    return latest_directory
