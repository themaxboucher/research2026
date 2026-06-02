import io
import logging
import tokenize
from pathlib import Path

from runs import require_latest_run_directory
from storage import load_from_json, save_to_json

ANALYSED_DATASET_FILENAME = "files_analysed"


def _extract_comment_tokens(file_content: str) -> list:
    return [
        tok
        for tok in tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline)
        if tok.type == tokenize.COMMENT
    ]


def count_loc(file_content: str) -> int:
    return len(file_content.splitlines())


def count_comments(file_content: str) -> int:
    return len(_extract_comment_tokens(file_content))


def avg_comment_character_length(file_content: str) -> float:
    comment_tokens = _extract_comment_tokens(file_content)
    if not comment_tokens:
        return 0.0
    total_character_count = sum(len(tok.string.rstrip("\r\n")) for tok in comment_tokens)
    return total_character_count / len(comment_tokens)


def analyse_file_content(file_content: str) -> dict:
    line_count = count_loc(file_content)
    comment_count = count_comments(file_content)
    comments_per_loc = comment_count / line_count if line_count else 0.0
    avg_comment_length = avg_comment_character_length(file_content)

    return {
        "loc": line_count,
        "comments": comment_count,
        "comments_per_loc": comments_per_loc,
        "avg_comment_length": avg_comment_length,
    }


def analyse_file_record(source_file: dict) -> None:
    if source_file.get("content"):
        source_file["original_metrics"] = analyse_file_content(source_file["content"])

    if source_file.get("generated_content"):
        source_file["generated_metrics"] = analyse_file_content(source_file["generated_content"])


def analyse_dataset(run_dir: Path) -> None:
    all_files = load_from_json(run_dir, "files_generated")
    total_files = len(all_files)
    logging.info("Analysing %d files...", total_files)

    succeeded_count = 0
    skipped_count = 0

    for index, source_file in enumerate(all_files, start=1):
        logging.info(
            "Analysing %s (%d/%d)...",
            source_file["filepath"],
            index,
            total_files,
        )
        try:
            analyse_file_record(source_file)
            succeeded_count += 1
        except ValueError as error:
            logging.warning("Skipping %s: %s", source_file["filepath"], error)
            skipped_count += 1

    save_to_json(all_files, run_dir, ANALYSED_DATASET_FILENAME)
    logging.info(
        "Saved enriched dataset to %s (%d succeeded, %d skipped)",
        run_dir / f"{ANALYSED_DATASET_FILENAME}.json",
        succeeded_count,
        skipped_count,
    )


if __name__ == "__main__":
    analyse_dataset(require_latest_run_directory())
