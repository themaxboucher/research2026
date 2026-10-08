import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from collect.constants import DATASET_FILENAME
from collect.filter_rules import CONTEXT_LIMIT_MODELS, drop_context_overflows
from storage import rewrite_jsonl
from storage.datasets import (
    MANIFEST_FILENAME,
    resolve_dataset_directory,
    write_manifest,
)


@dataclass
class ContextOverflowFilter:
    """Apply filter.py's last step to records it already kept, counting what
    remains the way filter.py counts it."""

    num_files_context_overflow: int = 0
    num_target_comments_context_overflow: int = 0
    num_files: int = 0
    num_target_comments: int = 0
    repo_names: set[str] = field(default_factory=set)
    commit_keys: set[tuple[str, str]] = field(default_factory=set)

    def filter_record(self, record: dict) -> dict | None:
        target_comments = record["target_comments"]
        fitting_target_comments = drop_context_overflows(target_comments)
        overflowing_comment_count = len(target_comments) - len(fitting_target_comments)
        self.num_target_comments_context_overflow += overflowing_comment_count
        if not fitting_target_comments:
            self.num_files_context_overflow += 1
            return None

        record["target_comments"] = fitting_target_comments
        self.num_target_comments += len(fitting_target_comments)
        self.num_files += 1
        self.repo_names.add(record["repo_name"])
        self.commit_keys.add((record["repo_name"], record["commit_hash"]))
        return record

    def manifest_counts(self) -> dict:
        return {
            "num_files_context_overflow": self.num_files_context_overflow,
            "num_repos": len(self.repo_names),
            "num_commits": len(self.commit_keys),
            "num_files": self.num_files,
            "num_target_comments": self.num_target_comments,
            "num_target_comments_context_overflow": (
                self.num_target_comments_context_overflow
            ),
        }


def _filter_context_overflows(dataset_directory: Path) -> None:
    """Drop the context overflows from a dataset filtered before filter.py had
    that step, leaving the dataset and its manifest as a fresh filter would."""
    manifest_path = dataset_directory / f"{MANIFEST_FILENAME}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "context_limit_models" in manifest:
        raise SystemExit(
            f"{dataset_directory.name} is already filtered for context overflows "
            f"({', '.join(manifest['context_limit_models'])})."
        )

    context_overflow_filter = ContextOverflowFilter()
    rewrite_jsonl(
        dataset_directory, DATASET_FILENAME, context_overflow_filter.filter_record
    )
    write_manifest(
        dataset_directory,
        {
            **manifest,
            **context_overflow_filter.manifest_counts(),
            "context_limit_models": CONTEXT_LIMIT_MODELS,
        },
    )

    logging.info(
        "Dropped %d target comments that overflow a model's context, %d files "
        "with them. Kept %d target comments in %d files",
        context_overflow_filter.num_target_comments_context_overflow,
        context_overflow_filter.num_files_context_overflow,
        context_overflow_filter.num_target_comments,
        context_overflow_filter.num_files,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Filtered dataset directory to drop the context overflows from "
        "(defaults to the latest dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = resolve_dataset_directory(args.dataset_dir)

    _filter_context_overflows(dataset_directory)


if __name__ == "__main__":
    main()
