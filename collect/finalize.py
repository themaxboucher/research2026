from pathlib import Path
import argparse
import logging

from storage import merge_jsonl_shards
from collect.constants import DATA_FILENAME, MINNED_REPOS_FILENAME


def _finalize(dataset_directory: Path) -> None:
    repo_file_shards = merge_jsonl_shards(
        dataset_directory, DATA_FILENAME, delete_shards=True
    )
    mined_repo_shards = merge_jsonl_shards(
        dataset_directory, MINNED_REPOS_FILENAME, delete_shards=True
    )
    logging.info(
        "Merged %d %s shards and %d %s shards",
        repo_file_shards,
        DATA_FILENAME,
        mined_repo_shards,
        MINNED_REPOS_FILENAME,
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="The dataset directory containing the JSONL shards",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    dataset_directory = Path(args.dataset_dir)
    
    _finalize(dataset_directory)


if __name__ == "__main__":
    main()
