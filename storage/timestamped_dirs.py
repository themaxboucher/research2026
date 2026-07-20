from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


def is_timestamp_name(name: str) -> bool:
    try:
        datetime.strptime(name, TIMESTAMP_FORMAT)
        return True
    except ValueError:
        return False


def timestamp_from_name(name: str) -> datetime:
    return datetime.strptime(name, TIMESTAMP_FORMAT)


def new_timestamped_directory(base_directory: Path) -> Path:
    directory = base_directory / datetime.now().strftime(TIMESTAMP_FORMAT)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_timestamped_argument(
    argument: str, base_directory: Path, *, label: str
) -> Path:
    """Resolve a CLI directory argument to a path. A bare timestamp name is
    resolved under `base_directory`; a path ending in a timestamp is used as-is.
    `label` names the directory kind in the error message (e.g. "run directory").
    """
    directory = Path(argument)
    if not is_timestamp_name(directory.name):
        raise SystemExit(
            f"Invalid {label} '{argument}': the directory name must be a "
            "timestamp in the format YYYY-MM-DDTHH-MM-SS"
        )
    if directory.parent == Path("."):
        return base_directory / directory.name
    return directory


def find_latest_timestamped_directory(base_directory: Path) -> Path | None:
    if not base_directory.exists():
        return None
    existing_directories = sorted(
        path for path in base_directory.iterdir() if path.is_dir()
    )
    if not existing_directories:
        return None
    return existing_directories[-1]
