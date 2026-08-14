import argparse
import logging
from pathlib import Path

from analyze.complexity import cognitive_complexity, cyclomatic_complexity
from generate.constants import GENERATE_FILENAME
from storage.jsonl import load_from_jsonl, save_to_jsonl
from storage.runs import resolve_dataset_and_run


def _analyze(run_dir: Path) -> None:
    filename = GENERATE_FILENAME + "_scored"
    records = load_from_jsonl(run_dir, filename)

    record_num = 0
    num_scored_records = 0

    for record in records:
        record_num += 1
        logging.info("Processing record %d", record_num)
        for comment_generation in record.get("comment_generations") or []:
            try:
                cognitive_comp = cognitive_complexity(comment_generation["code_scope"])
                comment_generation["cognitive_complexity"] = cognitive_comp
            except Exception as e:
                logging.warning(
                    "Error occurred while calculating cognitive complexity: %s", e
                )
            try:
                cyclomatic_comp = cyclomatic_complexity(comment_generation["code_scope"])
                comment_generation["cyclomatic_complexity"] = cyclomatic_comp
            except Exception as e:
                logging.warning(
                    "Error occurred while calculating cyclomatic complexity: %s", e
                )
            num_scored_records += 1

    save_to_jsonl(records, run_dir, filename)

    logging.info("Wrote %s.jsonl with complexity metrics", filename)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to evaluate for (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to evaluate (defaults to the latest run in the dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)

    _analyze(run_directory)


if __name__ == "__main__":
    main()
