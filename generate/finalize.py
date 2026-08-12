import argparse
import logging
from pathlib import Path

from generate.constants import (
    GENERATE_FILENAME,
    PROGRESS_FILENAME,
)
from storage.jsonl import iter_from_jsonl, merge_jsonl_shards, save_to_jsonl
from storage.runs import resolve_dataset_and_run


def _record_key(record: dict) -> tuple[str | None, str | None, str | None]:
    # Must stay in step with the key generate.py partitions and resumes on.
    return (record.get("repo_name"), record.get("new_path"), record.get("commit_hash"))


def _comment_key(comment_generation: dict) -> tuple[str | None, int | None, int | None]:
    # A file cannot hold two comments of the same type over the same line span.
    return (
        comment_generation.get("type"),
        comment_generation.get("start_line"),
        comment_generation.get("end_line"),
    )


def _regrouped_generation_records(run_dir: Path, shard_paths: list[Path]) -> list[dict]:
    """Collapse the per-(model, partition) shards into one record per dataset record."""
    regrouped: dict[tuple, dict] = {}
    comment_indexes: dict[tuple, dict[tuple, dict]] = {}

    # Sorted order is task order, which is model-major, so every record's
    # `results` come out in the same model order.
    for shard_path in shard_paths:
        for record in iter_from_jsonl(run_dir, shard_path.stem):
            record_key = _record_key(record)
            regrouped_record = regrouped.get(record_key)
            if regrouped_record is None:
                regrouped_record = {
                    "repo_name": record.get("repo_name"),
                    "commit_hash": record.get("commit_hash"),
                    "new_path": record.get("new_path"),
                    "comment_generations": [],
                }
                regrouped[record_key] = regrouped_record
                comment_indexes[record_key] = {}

            comment_index = comment_indexes[record_key]
            for comment_generation in record.get("comment_generations") or []:
                comment_key = _comment_key(comment_generation)
                regrouped_generation = comment_index.get(comment_key)

                # The first model to reach a comment brings the fields
                # shared across models (the prompt, the reference comment,
                # its location). Later models only add their results.
                if regrouped_generation is None:
                    regrouped_generation = dict(comment_generation)
                    regrouped_generation["results"] = list(
                        comment_generation.get("results") or []
                    )
                    comment_index[comment_key] = regrouped_generation
                    regrouped_record["comment_generations"].append(regrouped_generation)
                else:
                    regrouped_generation["results"].extend(
                        comment_generation.get("results") or []
                    )

    return list(regrouped.values())


def _finalize(run_dir: Path) -> None:
    # Regrouping holds the merged records in memory, so it reads the shards
    # before anything is deleted and writes the result in one go.
    generation_shards = sorted(run_dir.glob(f"{GENERATE_FILENAME}.*.jsonl"))
    if generation_shards:
        generation_records = _regrouped_generation_records(run_dir, generation_shards)
        save_to_jsonl(generation_records, run_dir, GENERATE_FILENAME)
        for shard_path in generation_shards:
            shard_path.unlink()
        logging.info(
            "Merged %d %s shards into %d records",
            len(generation_shards),
            GENERATE_FILENAME,
            len(generation_records),
        )

    # Progress stays one row per (model, dataset record). It is the resume
    # ledger, so it is concatenated rather than regrouped.
    if any(run_dir.glob(f"{PROGRESS_FILENAME}.*.jsonl")):
        shard_count = merge_jsonl_shards(run_dir, PROGRESS_FILENAME, delete_shards=True)
        logging.info("Merged %d %s shards", shard_count, PROGRESS_FILENAME)

    # Progress is recorded per (model, dataset record), so these count
    # generations rather than distinct records.
    completed_generations = 0
    failed_generations = 0
    if (run_dir / f"{PROGRESS_FILENAME}.jsonl").exists():
        for record in iter_from_jsonl(run_dir, PROGRESS_FILENAME):
            if record.get("error"):
                failed_generations += 1
            else:
                completed_generations += 1
    logging.info(
        "Finalized generation %r: %d generations completed, %d failed",
        run_dir.name,
        completed_generations,
        failed_generations,
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
