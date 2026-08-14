import argparse
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from generate.constants import GENERATE_FILENAME
from storage.jsonl import iter_from_jsonl
from storage.runs import resolve_dataset_and_run

MAX_REASON_LENGTH = 80
MISSING_ERROR_REASON = "No comment returned and no error recorded"
UNRECORDED_ERROR_REASON = "Failed with an empty error message"

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def failure_reason(result: dict) -> str:
    """Collapse one failed completion's error into the reason it is grouped
    under. Provider errors embed the specifics of the call that hit them --
    sizes, device ids, process ids -- so those are generalized away to leave
    the kind of failure behind."""
    error_message = result.get("error")
    if error_message is None:
        return MISSING_ERROR_REASON

    first_line = error_message.strip().split("\n")[0]
    single_spaced = _WHITESPACE_PATTERN.sub(" ", first_line).strip()
    if not single_spaced:
        return UNRECORDED_ERROR_REASON

    generalized = _NUMBER_PATTERN.sub("N", single_spaced)
    if len(generalized) <= MAX_REASON_LENGTH:
        return generalized
    return generalized[:MAX_REASON_LENGTH].rstrip() + "..."


def is_failed_completion(result: dict) -> bool:
    return result.get("error") is not None or result.get("comment_text") is None


@dataclass
class CompletionFailureTally:
    completion_count: int = 0
    failures_by_reason: Counter[str] = field(default_factory=Counter)

    def add_completion(self, result: dict) -> None:
        self.completion_count += 1
        if is_failed_completion(result):
            self.failures_by_reason[failure_reason(result)] += 1

    @property
    def failure_count(self) -> int:
        return sum(self.failures_by_reason.values())

    @property
    def failure_percentage(self) -> float:
        if self.completion_count == 0:
            return 0.0
        return 100.0 * self.failure_count / self.completion_count


def _completion_results(record: dict) -> Iterator[dict]:
    for comment_generation in record.get("comment_generations") or []:
        yield from comment_generation.get("results") or []


def tally_failed_completions(run_directory: Path) -> CompletionFailureTally:
    """Count every LLM completion in a finalized run and group the failed ones
    by the reason they failed."""
    tally = CompletionFailureTally()
    for record in iter_from_jsonl(run_directory, GENERATE_FILENAME):
        for result in _completion_results(record):
            tally.add_completion(result)
    return tally


def _failure_report_lines(
    run_directory: Path, tally: CompletionFailureTally
) -> list[str]:
    heading = f"Failed completions in {run_directory.name} ({GENERATE_FILENAME}.jsonl)"
    if tally.completion_count == 0:
        return [heading, "", "No completions recorded."]
    if tally.failure_count == 0:
        return [
            heading,
            "",
            f"No failures in {tally.completion_count:,} completions.",
        ]

    ranked_failures = tally.failures_by_reason.most_common()
    count_width = max(len(f"{count:,}") for _, count in ranked_failures)
    count_width = max(count_width, len(f"{tally.failure_count:,}"), len("Count"))

    lines = [heading, "", f"  {'Count':>{count_width}}   Share   Failure reason"]
    for reason, count in ranked_failures:
        share = 100.0 * count / tally.failure_count
        lines.append(f"  {count:>{count_width},}  {share:5.1f}%   {reason}")

    lines.append(f"  {'-' * count_width}")
    lines.append(
        f"  {tally.failure_count:>{count_width},}  100.0%   Total failed completions"
    )
    lines.append("")
    lines.append(
        f"{tally.failure_count:,} of {tally.completion_count:,} completions "
        f"failed ({tally.failure_percentage:.1f}%), "
        f"across {len(ranked_failures)} distinct reasons."
    )
    return lines


def print_failure_report(run_directory: Path, tally: CompletionFailureTally) -> None:
    print("\n".join(_failure_report_lines(run_directory, tally)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to audit (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to audit (defaults to the latest run in the dataset)",
    )
    return parser.parse_args()


def _generation_file_missing_message(run_directory: Path) -> str:
    missing = f"No {GENERATE_FILENAME}.jsonl in {run_directory}."
    if any(run_directory.glob(f"{GENERATE_FILENAME}.*.jsonl")):
        return f"{missing} Run generate.finalize to merge the shards first."
    return missing


def main():
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)

    if not (run_directory / f"{GENERATE_FILENAME}.jsonl").exists():
        raise SystemExit(_generation_file_missing_message(run_directory))

    print_failure_report(run_directory, tally_failed_completions(run_directory))


if __name__ == "__main__":
    main()
