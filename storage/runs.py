import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from storage.datasets import dataset_directory_from_argument, latest_dataset_directory
from storage.timestamped_dirs import (
    find_latest_timestamped_directory,
    new_timestamped_directory,
    resolve_timestamped_argument,
)

RUNS_DIRECTORY_NAME = "runs"
MANIFEST_FILENAME = "run.json"


def runs_root(dataset_directory: Path) -> Path:
    return dataset_directory / RUNS_DIRECTORY_NAME


def run_directory_from_argument(run_dir_argument: str, dataset_directory: Path) -> Path:
    return resolve_timestamped_argument(
        run_dir_argument, runs_root(dataset_directory), label="run directory"
    )


def create_new_run_directory(dataset_directory: Path) -> Path:
    return new_timestamped_directory(runs_root(dataset_directory))


def find_latest_run_directory(dataset_directory: Path) -> Path | None:
    return find_latest_timestamped_directory(runs_root(dataset_directory))


def latest_run_directory(dataset_directory: Path) -> Path:
    latest_directory = find_latest_run_directory(dataset_directory)
    if latest_directory is None:
        raise SystemExit(
            f"No run directory found under {runs_root(dataset_directory)}."
        )
    return latest_directory


def _current_git_commit() -> str | None:
    """Return the current git commit hash, or None if git isn't available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip() or None


def write_manifest(
    run_dir: Path,
    model_profile: str | None,
    model_names: list[str],
    config: dict,
    created_at: str | None = None,
) -> dict:
    manifest = {
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_profile": model_profile,
        "model_names": list(model_names),
        "git_commit": _current_git_commit(),
        "config": config,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def read_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def resolve_dataset_and_run(
    dataset_dir_arg: str | None,
    run_dir_arg: str | None,
    *,
    create_run: bool = False,
) -> tuple[Path, Path]:
    """Resolve the dataset and run directories from CLI arguments.

    The dataset defaults to the latest dataset. The run defaults to a new
    timestamped directory when `create_run` is set (preparation) or the latest
    existing run otherwise (generation, finalization). Both choices are logged.
    """
    if dataset_dir_arg:
        dataset_directory = dataset_directory_from_argument(dataset_dir_arg)
    else:
        dataset_directory = latest_dataset_directory()
    logging.info("Using dataset directory: %s", dataset_directory)

    if run_dir_arg:
        run_directory = run_directory_from_argument(run_dir_arg, dataset_directory)
    elif create_run:
        run_directory = create_new_run_directory(dataset_directory)
    else:
        run_directory = latest_run_directory(dataset_directory)
    logging.info("Using run directory: %s", run_directory)

    return dataset_directory, run_directory
