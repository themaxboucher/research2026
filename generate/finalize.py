import logging
from pathlib import Path

from generate.constants import (
    LOCATION_FILENAME,
    REGENERATE_FILENAME,
    PROGRESS_FILENAME,
)
from generate.generate import generation_dir
from storage.jsonl import merge_jsonl_shards, iter_from_jsonl

def _finalize(run_dir: Path, label: str) -> Path:
    """Merge the per-task shard files into the unsharded files readers expect."""
    gen_dir = generation_dir(run_dir, label)

    for filename in (LOCATION_FILENAME, REGENERATE_FILENAME, PROGRESS_FILENAME):
        # Skip filenames with no shards so an approach that never ran doesn't
        # get an empty merged file (which readers would take as real output).
        if not any(gen_dir.glob(f"{filename}.*.jsonl")):
            continue
        shard_count = merge_jsonl_shards(gen_dir, filename)
        logging.info("Merged %d %s shards", shard_count, filename)

    completed_files = 0
    failed_files = 0
    if (gen_dir / f"{PROGRESS_FILENAME}.jsonl").exists():
        for record in iter_from_jsonl(gen_dir, PROGRESS_FILENAME):
            if record.get("error"):
                failed_files += 1
            else:
                completed_files += 1
    logging.info(
        "Finalized generation %r: %d files generated, %d failed",
        label,
        completed_files,
        failed_files,
    )
    return gen_dir


def _parse_args():
    pass


def main():
    _finalize()


if __name__ == "__main__":
    main()