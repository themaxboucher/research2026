from datetime import datetime
from pathlib import Path

RUNS_DIRECTORY = Path(__file__).parent / "runs"
RUN_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


def create_new_run_directory() -> Path:
    new_run_directory = RUNS_DIRECTORY / datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    new_run_directory.mkdir(parents=True, exist_ok=True)
    return new_run_directory


def find_latest_run_directory() -> Path | None:
    if not RUNS_DIRECTORY.exists():
        return None

    existing_run_directories = sorted(
        path for path in RUNS_DIRECTORY.iterdir() if path.is_dir()
    )
    if not existing_run_directories:
        return None

    return existing_run_directories[-1]


def resolve_run_directory(create_new_run: bool) -> Path:
    if create_new_run:
        return create_new_run_directory()

    latest_run_directory = find_latest_run_directory()
    if latest_run_directory is not None:
        return latest_run_directory

    return create_new_run_directory()


def require_latest_run_directory() -> Path:
    latest_run_directory = find_latest_run_directory()
    if latest_run_directory is None:
        raise SystemExit(
            "No run directory found. Run `python main.py --collect` first."
        )
    return latest_run_directory
