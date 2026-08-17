import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from eval.constants import (
    LOCATION_METRICS_FILENAME,
    SCORE_METRICS,
    UNKNOWN_MODEL,
)
from eval.manifest import read_eval_manifest
from eval.normalize import normalize_comment
from eval.scorer import CommentScorer
from generate.constants import GENERATE_FILENAME
from storage import load_from_jsonl, merge_jsonl_shards
from storage.runs import resolve_dataset_and_run


def _summarize_scores(
    scored: list[tuple[str, str, dict]], scorer: CommentScorer
) -> dict:
    predictions = [prediction for prediction, _, _ in scored]
    references = [reference for _, reference, _ in scored]
    summary = {
        "count": len(scored),
        **scorer.corpus_bleu(predictions, references),
    }
    for metric in SCORE_METRICS:
        values = [scores[metric] for _, _, scores in scored]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_median"] = statistics.median(values)
    return summary


def _save_metrics(metrics: list[dict], run_dir: Path, filename: str) -> None:
    path = run_dir / (filename + ".json")
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _iter_scored_location(records: list[dict]):
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            reference = normalize_comment(comment_generation.get("comment") or "")
            for result in comment_generation.get("results") or []:
                scores = result.get("scores")
                if scores is None:
                    continue
                yield (
                    result.get("model") or UNKNOWN_MODEL,
                    normalize_comment(result.get("comment_text") or ""),
                    reference,
                    scores,
                )


def _write_location_metrics(
    records: list[dict], run_dir: Path, scorer: CommentScorer
) -> None:
    scored_by_model = defaultdict(list)
    for model, prediction, reference, scores in _iter_scored_location(records):
        scored_by_model[model].append((prediction, reference, scores))

    metrics = [
        {"model": model, **_summarize_scores(scored_by_model[model], scorer)}
        for model in sorted(scored_by_model)
    ]
    _save_metrics(metrics, run_dir, LOCATION_METRICS_FILENAME)


def _merge_shards(run_dir: Path, filename: str, num_tasks: int | None) -> int:
    scored_filename = filename + "_scored"
    shard_count = len(list(run_dir.glob(f"{scored_filename}.*.jsonl")))
    # No shards means either an approach that never ran or a re-finalize after
    # an earlier merge already consumed them. Both leave the merged file alone.
    if shard_count == 0:
        return 0

    if num_tasks is None:
        raise SystemExit(
            f"Found {shard_count} {scored_filename} shard(s) in {run_dir.name} "
            "but no eval manifest to check them against; run eval.prepare first"
        )
    if shard_count != num_tasks:
        raise SystemExit(
            f"Found {shard_count} {scored_filename} shard(s) in {run_dir.name} "
            f"but the run was prepared for {num_tasks} task(s). Merging now "
            f"would overwrite {scored_filename}.jsonl with part of the run. "
            f"Score the missing tasks (--array 0-{num_tasks - 1} reruns all of "
            "them) before finalizing."
        )

    merge_jsonl_shards(run_dir, scored_filename, delete_shards=True)
    logging.info("Merged %d %s shards", shard_count, filename)
    return shard_count


def _finalize(run_dir: Path) -> None:
    """Merge the per-task scored shards, then write the aggregate metrics."""
    num_tasks = read_eval_manifest(run_dir).get("num_tasks")
    _merge_shards(run_dir, GENERATE_FILENAME, num_tasks)

    scorer = CommentScorer()

    if (run_dir / f"{GENERATE_FILENAME}_scored.jsonl").exists():
        records = load_from_jsonl(run_dir, (GENERATE_FILENAME + "_scored"))
        _write_location_metrics(records, run_dir, scorer)
        logging.info("Wrote location metrics for %s", run_dir.name)


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
    _finalize(run_directory)


if __name__ == "__main__":
    main()
