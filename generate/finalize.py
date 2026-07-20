import logging
import argparse
from pathlib import Path

from generate.constants import (
    LOCATION_FILENAME,
    REGENERATE_FILENAME,
    PROGRESS_FILENAME,
)
from generate.runs import resolve_dataset_and_run
from storage.jsonl import merge_jsonl_shards, iter_from_jsonl


def _finalize(run_dir: Path) -> None:
    """Merge the per-task shard files into the unsharded files readers expect."""
    # Shards are written directly into the run directory (the flat layout has no
    # per-label generation subdirectory), so merge them there.

    for filename in (LOCATION_FILENAME, REGENERATE_FILENAME, PROGRESS_FILENAME):
        # Skip filenames with no shards so an approach that never ran doesn't
        # get an empty merged file (which readers would take as real output).
        if not any(run_dir.glob(f"{filename}.*.jsonl")):
            continue
        shard_count = merge_jsonl_shards(run_dir, filename, delete_shards=True)
        logging.info("Merged %d %s shards", shard_count, filename)

    completed_files = 0
    failed_files = 0
    if (run_dir / f"{PROGRESS_FILENAME}.jsonl").exists():
        for record in iter_from_jsonl(run_dir, PROGRESS_FILENAME):
            if record.get("error"):
                failed_files += 1
            else:
                completed_files += 1
    logging.info(
        "Finalized generation %r: %d files generated, %d failed",
        run_dir.name,
        completed_files,
        failed_files,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to generate for (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to generate for (defaults to the latest run in the dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)

    _finalize(run_directory)


if __name__ == "__main__":
    main()
