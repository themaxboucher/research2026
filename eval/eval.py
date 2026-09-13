import argparse
import itertools
import logging
from pathlib import Path

from eval.manifest import read_eval_manifest
from eval.normalize import normalize_comment
from eval.scorer import CommentScorer
from generate.constants import GENERATE_FILENAME
from storage import iter_from_jsonl, save_to_jsonl, shard_suffix
from storage.runs import resolve_dataset_and_run


def _record_key(record: dict) -> tuple[str | None, str | None, str | None]:
    return (record.get("repo_name"), record.get("new_path"), record.get("commit_hash"))


def _result_key(record_key: tuple, comment_generation: dict, result: dict) -> tuple:
    # A comment gets one result per model, so the model identifies the result.
    return (
        *record_key,
        comment_generation.get("type"),
        comment_generation.get("start_line"),
        comment_generation.get("end_line"),
        result.get("model"),
    )


def _previous_scores(run_dir: Path, records: list[dict]) -> dict[tuple, tuple[str, dict]]:
    """Map each result of `records` scored in a previous pass to its (comment_text, scores)."""
    scored_filename = GENERATE_FILENAME + "_scored"
    if not (run_dir / f"{scored_filename}.jsonl").exists():
        return {}

    # The scored file is far larger than one stride and ordered differently,
    # so it is streamed and only this stride's records are kept.
    stride_keys = {_record_key(record) for record in records}
    previous_scores = {}
    for record in iter_from_jsonl(run_dir, scored_filename):
        record_key = _record_key(record)
        if record_key not in stride_keys:
            continue
        for comment_generation in record.get("comment_generations") or []:
            for result in comment_generation.get("results") or []:
                if result.get("scores") is None:
                    continue
                result_key = _result_key(record_key, comment_generation, result)
                previous_scores[result_key] = (result.get("comment_text"), result["scores"])
    return previous_scores


def _collect_pairs(
    records: list[dict], previous_scores: dict[tuple, tuple[str, dict]]
) -> list[tuple[dict, str, str]]:
    pending = []
    for record in records:
        record_key = _record_key(record)
        for comment_generation in record.get("comment_generations") or []:
            reference = normalize_comment(comment_generation.get("comment") or "")
            for result in comment_generation.get("results") or []:
                prediction = normalize_comment(result.get("comment_text") or "")
                if result.get("error") or not prediction or not reference:
                    result["scores"] = None
                    continue
                # A regenerated comment keeps its key but not its text, so an
                # old score is only reused when the text is unchanged.
                result_key = _result_key(record_key, comment_generation, result)
                previous = previous_scores.get(result_key)
                if previous is not None and previous[0] == result.get("comment_text"):
                    result["scores"] = previous[1]
                    continue
                pending.append((result, prediction, reference))
    return pending


def score_records(
    records: list[dict],
    scorer: CommentScorer,
    previous_scores: dict[tuple, tuple[str, dict]],
) -> int:
    pending = _collect_pairs(records, previous_scores)
    if pending:
        predictions = [prediction for _, prediction, _ in pending]
        references = [reference for _, _, reference in pending]
        for (result, _, _), scores in zip(
            pending,
            scorer.score_pairs(predictions, references, desc="Scoring location"),
        ):
            result["scores"] = scores
    return len(pending)


def _score_shard(run_dir: Path, task_id: int, num_tasks: int, force: bool) -> None:
    scorer = CommentScorer()
    suffix = shard_suffix(task_id, num_tasks)

    if not (run_dir / f"{GENERATE_FILENAME}.jsonl").exists():
        logging.warning("No %s.jsonl found. Nothing to score.", GENERATE_FILENAME)
        return

    # Striding deterministically partitions records across tasks
    shard_records = list(
        itertools.islice(iter_from_jsonl(run_dir, GENERATE_FILENAME), task_id, None, num_tasks)
    )
    previous_scores = {} if force else _previous_scores(run_dir, shard_records)
    num_scored = score_records(shard_records, scorer, previous_scores)
    # Write the full stride (scored and unusable alike) so finalize's merge
    # reconstructs every record, not just the ones scored this pass.
    save_to_jsonl(shard_records, run_dir, f"{GENERATE_FILENAME + '_scored'}.{suffix}")
    logging.info(
        "Task %d: scored %d new %s results across %d records",
        task_id,
        num_scored,
        GENERATE_FILENAME,
        len(shard_records),
    )


def _valid_manifest(run_dir: Path, task_id: int) -> dict:
    manifest = read_eval_manifest(run_dir)
    num_tasks = manifest.get("num_tasks")
    if num_tasks is None:
        raise RuntimeError(
            f"Run {run_dir.name} was not prepared for a scoring array; "
            "run eval.prepare first"
        )
    if not 0 <= task_id < num_tasks:
        raise ValueError(
            f"--task-id {task_id} is out of range for --num-tasks {num_tasks}"
        )
    return manifest


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
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the scoring array. Scores only this task's "
        "share of the records into its own sharded files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rescore every result instead of reusing scores from the previous "
        f"{GENERATE_FILENAME}_scored.jsonl",
    )
    args = parser.parse_args()
    if args.task_id is None:
        raise SystemExit(
            "--task-id is required: it is this task's index in the scoring array"
        )
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)
    manifest = _valid_manifest(run_directory, args.task_id)

    _score_shard(
        run_directory,
        task_id=args.task_id,
        num_tasks=manifest["num_tasks"],
        force=args.force,
    )


if __name__ == "__main__":
    main()
