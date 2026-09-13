from pathlib import Path

from generate.constants import GENERATE_FILENAME, PROGRESS_FILENAME
from storage import (
    drop_trailing_records,
    iter_from_jsonl,
    shard_filename,
    truncate_broken_tail,
)


def dataset_record_key(record: dict) -> tuple[str | None, str | None, str | None]:
    # The key generation partitions, resumes and regroups on.
    return (record.get("repo_name"), record.get("new_path"), record.get("commit_hash"))


def repair_interrupted_shard(
    run_dir: Path, suffix: str
) -> set[tuple[str | None, str | None, str | None]]:
    """Undo writes cut off by a killed task and return the records its progress shard committed."""
    progress_filename = shard_filename(PROGRESS_FILENAME, suffix)
    output_filename = shard_filename(GENERATE_FILENAME, suffix)

    # The progress shard is fixed first, since reading its keys parses every line
    truncate_broken_tail(run_dir, progress_filename)
    completed_keys = set()
    if (run_dir / f"{progress_filename}.jsonl").exists():
        completed_keys = {
            dataset_record_key(record)
            for record in iter_from_jsonl(run_dir, progress_filename)
        }

    # An output record is written before its progress row, so a kill between
    # the two can only leave uncommitted records at the end of the output shard.
    truncate_broken_tail(run_dir, output_filename)
    drop_trailing_records(
        run_dir,
        output_filename,
        lambda record: dataset_record_key(record) not in completed_keys,
    )
    return completed_keys
