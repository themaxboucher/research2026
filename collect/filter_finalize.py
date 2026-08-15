import argparse
import json
import logging
from pathlib import Path

from collect.constants import DATASET_FILENAME
from storage import merge_jsonl_shards
from storage.datasets import (
    MANIFEST_FILENAME,
    resolve_dataset_directory,
    write_manifest,
)


def _read_shard_manifests(dataset_directory: Path) -> list[dict]:
    # A task writes its manifest last, so a shard manifest means that task ran to completion.
    return [
        json.loads(manifest_path.read_text(encoding="utf-8"))
        for manifest_path in sorted(
            dataset_directory.glob(f"{MANIFEST_FILENAME}.*.json")
        )
    ]


def _validated_num_tasks(shard_manifests: list[dict], dataset_directory: Path) -> int:
    array_widths = {manifest["num_tasks"] for manifest in shard_manifests}
    if len(array_widths) > 1:
        raise SystemExit(
            f"Filter shards in {dataset_directory.name} disagree on the array "
            f"width ({sorted(array_widths)}). Each width partitions the commits "
            "differently, so the shards are not one dataset. Delete them and "
            "refilter with a single --num-tasks."
        )

    num_tasks = array_widths.pop()
    missing_task_ids = sorted(
        set(range(num_tasks)) - {manifest["task_id"] for manifest in shard_manifests}
    )
    if missing_task_ids:
        raise SystemExit(
            f"Missing filter shards for task(s) "
            f"{','.join(str(task_id) for task_id in missing_task_ids)} of "
            f"{num_tasks} in {dataset_directory.name}. Merging now would drop "
            f"their commits from {DATASET_FILENAME}.jsonl. Refilter them with "
            f"--num-tasks {num_tasks} --array "
            f"{','.join(str(task_id) for task_id in missing_task_ids)} first."
        )

    return num_tasks


REPO_COUNTS_BY_NAMES_FIELD = {
    "raw_repo_names": "raw_num_repos",
    "repo_names": "num_repos",
}


def _merged_counts(shard_manifests: list[dict]) -> dict:
    # Every other count is over files, comments or whole commits, none of which
    # a task shares with another, so they add up.
    merged: dict[str, int] = {}
    for manifest in shard_manifests:
        for key, value in manifest["counts"].items():
            merged[key] = merged.get(key, 0) + value

    for names_field, count_key in REPO_COUNTS_BY_NAMES_FIELD.items():
        distinct_repos: set[str] = set()
        for manifest in shard_manifests:
            distinct_repos.update(manifest[names_field])
        merged[count_key] = len(distinct_repos)

    return merged


def _finalize(dataset_directory: Path) -> None:
    """Merge the per-task filter shards and their manifests into the dataset."""
    shard_manifests = _read_shard_manifests(dataset_directory)
    if not shard_manifests:
        if (dataset_directory / f"{DATASET_FILENAME}.jsonl").exists():
            logging.info(
                "No filter shards left in %s; it is already finalized",
                dataset_directory.name,
            )
            return
        raise SystemExit(
            f"No filter shards found in {dataset_directory}. Run the filtering "
            "array (collect/scripts/filter-submit.sh) first."
        )

    num_tasks = _validated_num_tasks(shard_manifests, dataset_directory)

    # Checked before merging, because merging truncates the merged file first.
    dataset_shards = sorted(dataset_directory.glob(f"{DATASET_FILENAME}.*.jsonl"))
    if len(dataset_shards) != num_tasks:
        raise SystemExit(
            f"Found {len(dataset_shards)} {DATASET_FILENAME} shard(s) in "
            f"{dataset_directory.name} but {num_tasks} completed task(s). "
            "Refilter the array before finalizing."
        )

    merge_jsonl_shards(dataset_directory, DATASET_FILENAME, delete_shards=True)
    write_manifest(dataset_directory, _merged_counts(shard_manifests))
    for manifest_path in dataset_directory.glob(f"{MANIFEST_FILENAME}.*.json"):
        manifest_path.unlink()

    logging.info("Merged %d %s shards and their manifests", num_tasks, DATASET_FILENAME)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory holding the filter shards (defaults to the "
        "latest dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory = resolve_dataset_directory(args.dataset_dir)

    _finalize(dataset_directory)


if __name__ == "__main__":
    main()
