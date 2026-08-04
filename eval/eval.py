import argparse
import itertools
import logging
from pathlib import Path

from eval.manifest import read_eval_manifest
from eval.normalize import normalize_comment
from eval.scorer import CommentScorer
from generate.constants import LOCATION_FILENAME, REGENERATE_FILENAME
from storage import iter_from_jsonl, save_to_jsonl
from storage.runs import resolve_dataset_and_run


def _shard_suffix(task_id: int, num_tasks: int) -> str:
    digit_width = max(len(str(num_tasks - 1)), 1)
    return f"{task_id:0{digit_width}d}"


def _collect_pairs(records: list[dict]) -> list[tuple[dict, str, str]]:
    pending = []
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            reference = normalize_comment(comment_generation.get("comment") or "")
            for result in comment_generation.get("results") or []:
                prediction = normalize_comment(result.get("comment_text") or "")
                if result.get("error") or not prediction or not reference:
                    result["scores"] = None
                    continue
                pending.append((result, prediction, reference))
    return pending


def _iter_valid_extractions(records: list[dict]):
    for record in records:
        targets = record.get("targets") or []
        for result in record.get("results") or []:
            if result.get("error"):
                continue
            for target, extraction in zip(targets, result.get("extractions") or []):
                if extraction.get("error"):
                    continue
                yield result, target, extraction


def _collect_pairs_regenerations(
    records: list[dict],
) -> list[tuple[dict, str, str]]:
    pending = []
    for _, target, extraction in _iter_valid_extractions(records):
        reference = normalize_comment(target.get("comment") or "")
        prediction = normalize_comment(extraction.get("comment_text") or "")
        if not extraction.get("placement_hit") or not prediction or not reference:
            extraction["scores"] = None
            continue
        pending.append((extraction, prediction, reference))
    return pending


def score_location_records(records: list[dict], scorer: CommentScorer) -> int:
    pending = _collect_pairs(records)
    if pending:
        predictions = [prediction for _, prediction, _ in pending]
        references = [reference for _, _, reference in pending]
        for (result, _, _), scores in zip(
            pending,
            scorer.score_pairs(predictions, references, desc="Scoring location"),
        ):
            result["scores"] = scores
    return len(pending)


def score_regeneration_records(records: list[dict], scorer: CommentScorer) -> int:
    pending = _collect_pairs_regenerations(records)
    if pending:
        predictions = [prediction for _, prediction, _ in pending]
        references = [reference for _, _, reference in pending]
        for (extraction, _, _), scores in zip(
            pending,
            scorer.score_pairs(predictions, references, desc="Scoring regenerate"),
        ):
            extraction["scores"] = scores
    return len(pending)


def _score_shard(run_dir: Path, task_id: int, num_tasks: int) -> None:
    # BERTScore is loaded lazily, so a shard with nothing scorable (every
    # result errored or came back empty) never pays to load the model.
    scorer = CommentScorer()
    suffix = _shard_suffix(task_id, num_tasks)

    approaches = (
        (LOCATION_FILENAME, score_location_records),
        (REGENERATE_FILENAME, score_regeneration_records),
    )
    for filename, score_records in approaches:
        if not (run_dir / f"{filename}.jsonl").exists():
            continue

        # Striding deterministically partitions records across tasks, so each
        # task materializes only its 1/num_tasks share rather than the whole run.
        shard_records = list(
            itertools.islice(
                iter_from_jsonl(run_dir, filename), task_id, None, num_tasks
            )
        )
        num_scored = score_records(shard_records, scorer)
        # Write the full stride (scored and unusable alike) so finalize's merge
        # reconstructs every record, not just the ones scored this pass.
        save_to_jsonl(shard_records, run_dir, f"{filename + '_scored'}.{suffix}")
        logging.info(
            "Task %d: scored %d new %s results across %d records",
            task_id,
            num_scored,
            filename,
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
    )


if __name__ == "__main__":
    main()
