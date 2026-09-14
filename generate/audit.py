import argparse
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from generate.constants import GENERATE_FILENAME, PROGRESS_FILENAME, SOURCE_FILENAME
from generate.shards import dataset_record_key
from storage.jsonl import iter_from_jsonl, shard_filename, shard_suffix
from storage.runs import read_manifest, resolve_dataset_and_run

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


def comment_generation_succeeded(comment_generation: dict) -> bool:
    results = comment_generation.get("results") or []
    return bool(results) and not any(is_failed_completion(result) for result in results)


def _record_has_failed_generation(record: dict) -> bool:
    comment_generations = record.get("comment_generations") or []
    return not all(
        comment_generation_succeeded(comment_generation)
        for comment_generation in comment_generations
    )


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


def _eligible_record_count(dataset_directory: Path, max_generate: int | None) -> int:
    source_path = dataset_directory / f"{SOURCE_FILENAME}.jsonl"
    with source_path.open("r", encoding="utf-8") as source_file:
        record_count = sum(1 for line in source_file if line.strip())
    if max_generate is None:
        return record_count
    return min(record_count, max_generate)


def _task_needs_generation(
    run_directory: Path, task_id: int, array_size: int, partition_record_count: int
) -> bool:
    suffix = shard_suffix(task_id, array_size)
    progress_filename = shard_filename(PROGRESS_FILENAME, suffix)
    if not (run_directory / f"{progress_filename}.jsonl").exists():
        return True

    committed_keys = {
        dataset_record_key(progress_row)
        for progress_row in iter_from_jsonl(run_directory, progress_filename)
    }
    if len(committed_keys) < partition_record_count:
        return True

    output_filename = shard_filename(GENERATE_FILENAME, suffix)
    return any(
        _record_has_failed_generation(record)
        for record in iter_from_jsonl(run_directory, output_filename)
    )


def task_ids_needing_generation(
    dataset_directory: Path, run_directory: Path
) -> list[int]:
    """Return the array tasks that have unfinished records or a failed comment
    generation: the tasks a --retry-failed resubmission has to run."""
    manifest = read_manifest(run_directory)
    if not manifest:
        raise SystemExit(f"No run manifest in {run_directory}.")

    config = manifest.get("config") or {}
    num_partitions = config["num_partitions"]
    array_size = num_partitions * len(manifest["model_names"])
    eligible_record_count = _eligible_record_count(
        dataset_directory, config.get("max_generate")
    )

    task_ids = []
    for task_id in range(array_size):
        partition = task_id % num_partitions
        partition_record_count = len(
            range(partition, eligible_record_count, num_partitions)
        )
        if _task_needs_generation(
            run_directory, task_id, array_size, partition_record_count
        ):
            task_ids.append(task_id)
    return task_ids


def array_spec(task_ids: list[int]) -> str:
    """Format task ids as a Slurm --array spec, collapsing consecutive ids into ranges."""
    id_ranges: list[list[int]] = []
    for task_id in task_ids:
        extends_previous_range = id_ranges and id_ranges[-1][1] == task_id - 1
        if extends_previous_range:
            id_ranges[-1][1] = task_id
        else:
            id_ranges.append([task_id, task_id])
    return ",".join(
        str(first_id) if first_id == last_id else f"{first_id}-{last_id}"
        for first_id, last_id in id_ranges
    )


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
    parser.add_argument(
        "--task-ids",
        action="store_true",
        help="Instead of the failure report, print the array tasks that have "
        "unfinished records or failed comment generations as a --array spec for "
        "submit.sh --retry-failed. Reads every shard, so run it while no generate "
        "task is writing to the run",
    )
    return parser.parse_args()


def _generation_file_missing_message(run_directory: Path) -> str:
    missing = f"No {GENERATE_FILENAME}.jsonl in {run_directory}."
    if any(run_directory.glob(f"{GENERATE_FILENAME}.*.jsonl")):
        return f"{missing} Run generate.finalize to merge the shards first."
    return missing


def main():
    args = _parse_args()

    dataset_directory, run_directory = resolve_dataset_and_run(
        args.dataset_dir, args.run_dir
    )

    if args.task_ids:
        task_ids = task_ids_needing_generation(dataset_directory, run_directory)
        if not task_ids:
            raise SystemExit("No tasks have unfinished records or failed generations.")
        print(array_spec(task_ids))
        return

    if not (run_directory / f"{GENERATE_FILENAME}.jsonl").exists():
        raise SystemExit(_generation_file_missing_message(run_directory))

    print_failure_report(run_directory, tally_failed_completions(run_directory))


if __name__ == "__main__":
    main()
