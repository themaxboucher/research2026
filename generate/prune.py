import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from generate.constants import GENERATE_FILENAME, SOURCE_FILENAME
from generate.shards import comment_key, dataset_record_key
from storage import iter_from_jsonl, rewrite_jsonl
from storage.runs import resolve_dataset_and_run

PRUNED_FILENAMES = (GENERATE_FILENAME, GENERATE_FILENAME + "_scored")


def _dataset_comment_keys(dataset_dir: Path) -> dict[tuple, set[tuple]]:
    """Map every file in the dataset to the keys of its target comments."""
    comment_keys_by_record = {}
    for record in iter_from_jsonl(dataset_dir, SOURCE_FILENAME):
        comment_keys_by_record[dataset_record_key(record)] = {
            comment_key(target_comment)
            for target_comment in record.get("target_comments") or []
        }
    return comment_keys_by_record


@dataclass
class RunPruner:
    dataset_comment_keys: dict[tuple, set[tuple]]
    num_records_dropped: int = 0
    num_comment_generations_dropped: int = 0

    def prune_record(self, record: dict) -> dict | None:
        comment_generations = record.get("comment_generations") or []
        kept_comment_keys = self.dataset_comment_keys.get(dataset_record_key(record))
        if kept_comment_keys is None:
            self.num_records_dropped += 1
            self.num_comment_generations_dropped += len(comment_generations)
            return None

        kept_generations = [
            comment_generation
            for comment_generation in comment_generations
            if comment_key(comment_generation) in kept_comment_keys
        ]
        dropped_generation_count = len(comment_generations) - len(kept_generations)
        self.num_comment_generations_dropped += dropped_generation_count
        record["comment_generations"] = kept_generations
        return record


def _prune(dataset_dir: Path, run_dir: Path) -> None:
    """Drop the generations of files and comments no longer in the dataset, so
    a run generated before the dataset was filtered further matches a run
    generated after.

    Only the merged files are pruned. The generation shards and run_progress.jsonl
    are left as they are, so the pruned run is finished: rerunning generate.finalize
    would bring the pruned generations back, and resuming or --retry-failed would
    partition the smaller dataset differently from the run.
    """
    dataset_comment_keys = _dataset_comment_keys(dataset_dir)
    logging.info(
        "Dataset %s has %d files to keep generations for",
        dataset_dir.name,
        len(dataset_comment_keys),
    )

    for filename in PRUNED_FILENAMES:
        if not (run_dir / f"{filename}.jsonl").exists():
            logging.info("No %s.jsonl in %s, skipping", filename, run_dir.name)
            continue

        run_pruner = RunPruner(dataset_comment_keys)
        rewrite_jsonl(run_dir, filename, run_pruner.prune_record)
        logging.info(
            "Pruned %s.jsonl: dropped %d records and %d comment generations",
            filename,
            run_pruner.num_records_dropped,
            run_pruner.num_comment_generations_dropped,
        )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to prune the run down to (defaults to the latest "
        "dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to prune (defaults to the latest run in the dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory, run_directory = resolve_dataset_and_run(
        args.dataset_dir, args.run_dir
    )

    _prune(dataset_directory, run_directory)


if __name__ == "__main__":
    main()
