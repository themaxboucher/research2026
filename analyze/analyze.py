import argparse
import logging
from pathlib import Path

from analyze.complexity import (
    cognitive_complexity,
    cyclomatic_complexity,
    lines_of_code,
    logical_lines_of_code,
    prompt_comment_density,
)
from generate.constants import GENERATE_FILENAME
from storage.jsonl import load_from_jsonl, save_to_jsonl
from storage.runs import resolve_dataset_and_run

PROMPT_CODE_METRICS = {
    "cognitive_complexity": cognitive_complexity,
    "cyclomatic_complexity": cyclomatic_complexity,
    "lines_of_code": lines_of_code,
    "logical_lines_of_code": logical_lines_of_code,
    "comment_density": prompt_comment_density,
}


def _add_prompt_code_metrics(comment_generation: dict) -> None:
    for metric_name, compute_metric in PROMPT_CODE_METRICS.items():
        try:
            comment_generation[metric_name] = compute_metric(
                comment_generation["prompt_code"]
            )
        except Exception as e:
            logging.warning("Error occurred while calculating %s: %s", metric_name, e)


def _analyze(run_dir: Path) -> None:
    filename = GENERATE_FILENAME + "_scored"
    records = load_from_jsonl(run_dir, filename)

    record_num = 0

    for record in records:
        record_num += 1
        logging.info("Processing record %d", record_num)
        for comment_generation in record.get("comment_generations") or []:
            _add_prompt_code_metrics(comment_generation)

    save_to_jsonl(records, run_dir, filename)

    logging.info("Wrote %s.jsonl with code metrics", filename)


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
