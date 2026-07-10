from github import get_commit_message
from runs import require_latest_run_directory
from sample import SAMPLE_FILENAME
from storage import load_from_jsonl, save_to_jsonl
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import logging

MESSAGE_FIELD = "commit_message"


def _record_needs_message(record: dict) -> bool:
    return record.get(MESSAGE_FIELD) is None


def _commits_needing_messages(records: list[dict]) -> list[tuple[str, str]]:
    commits: list[tuple[str, str]] = []
    seen_commits: set[tuple[str, str]] = set()
    for record in records:
        if not _record_needs_message(record):
            continue
        commit = (record["repo_name"], record["commit_hash"])
        if commit in seen_commits:
            continue
        seen_commits.add(commit)
        commits.append(commit)
    return commits


def _fetch_commit_messages(
    commits: list[tuple[str, str]],
) -> dict[tuple[str, str], str | None]:
    messages: dict[tuple[str, str], str | None] = {}
    progress_bar = tqdm(commits, desc="Fetching commit messages", unit="commit")
    for repo_name, commit_hash in progress_bar:
        try:
            messages[(repo_name, commit_hash)] = get_commit_message(
                repo_name, commit_hash
            )
        except RuntimeError as error:
            logging.warning(
                "Could not fetch commit message for %s@%s: %s",
                repo_name,
                commit_hash,
                error,
            )
            messages[(repo_name, commit_hash)] = None
    return messages


def _save_atomically(records: list[dict], run_dir: Path, filename: str) -> None:
    temp_filename = f"{filename}.tmp"
    save_to_jsonl(records, run_dir, temp_filename)
    temp_path = run_dir / f"{temp_filename}.jsonl"
    temp_path.replace(run_dir / f"{filename}.jsonl")


def enrich_sample_with_commit_messages(run_dir: Path) -> int:
    records = load_from_jsonl(run_dir, SAMPLE_FILENAME)
    commits = _commits_needing_messages(records)
    if not commits:
        logging.info(
            "All %d records in %s.jsonl already have a commit message",
            len(records),
            SAMPLE_FILENAME,
        )
        return 0

    messages = _fetch_commit_messages(commits)
    enriched_records = 0
    for record in records:
        if not _record_needs_message(record):
            continue
        record[MESSAGE_FIELD] = messages[(record["repo_name"], record["commit_hash"])]
        if record[MESSAGE_FIELD] is not None:
            enriched_records += 1

    _save_atomically(records, run_dir, SAMPLE_FILENAME)

    failed_commits = sum(1 for message in messages.values() if message is None)
    if failed_commits:
        logging.warning(
            "Could not fetch %d of %d commit messages; their records keep a "
            "null %s and will be retried on the next run",
            failed_commits,
            len(commits),
            MESSAGE_FIELD,
        )
    logging.info(
        "Enriched %d of %d records in %s.jsonl",
        enriched_records,
        len(records),
        SAMPLE_FILENAME,
    )
    return enriched_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Add a {MESSAGE_FIELD} field to every record of "
        f"{SAMPLE_FILENAME}.jsonl by fetching each commit from the GitHub API"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to enrich (defaults to the latest run)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else require_latest_run_directory()
    logging.info("Enriching sample in run directory: %s", run_dir)
    enrich_sample_with_commit_messages(run_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
