import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import evaluate

from generations import GENERATED_FILENAME, list_generations
from runs import require_latest_run_directory
from storage import load_from_jsonl

METRIC_NAMES = ("bleu4", "rougeL", "bertscore_f1")


def normalize_comment(text: str) -> str:
    """Comment text reduced to its content: per-line '#' markers and
    indentation stripped, whitespace runs collapsed to single spaces."""
    words = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        words.extend(stripped.split())
    return " ".join(words)


class CommentScorer:
    """Computes the three metrics over batches of (prediction, reference)
    pairs. Metrics are loaded lazily so runs with nothing left to score
    don't pay for loading the BERTScore model."""

    def __init__(self):
        self._bleu = None
        self._rouge = None
        self._bertscore = None

    def _load(self):
        if self._bleu is None:
            self._bleu = evaluate.load("bleu")
            self._rouge = evaluate.load("rouge")
            self._bertscore = evaluate.load("bertscore")

    def score_pairs(
        self, predictions: list[str], references: list[str]
    ) -> list[dict]:
        self._load()

        # BLEU is corpus-level, so compute it one pair at a time. Sentence-level
        # BLEU-4 needs smoothing, otherwise any pair without a matching 4-gram
        # scores exactly 0.
        bleu_scores = [
            self._bleu.compute(
                predictions=[pred], references=[ref], max_order=4, smooth=True
            )["bleu"]
            for pred, ref in zip(predictions, references)
        ]
        rouge_scores = self._rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=["rougeL"],
            use_aggregator=False,
        )["rougeL"]
        bertscore_f1 = self._bertscore.compute(
            predictions=predictions, references=references, lang="en"
        )["f1"]

        return [
            {"bleu4": bleu, "rougeL": rouge, "bertscore_f1": bert}
            for bleu, rouge, bert in zip(bleu_scores, rouge_scores, bertscore_f1)
        ]


def _collect_unscored(records: list[dict], force: bool) -> list[tuple[dict, str, str]]:
    """Results that still need scoring, as (result, prediction, reference)
    with prediction/reference already normalized. Unusable results get
    `scores: null` right away and are not returned."""
    pending = []
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            reference = normalize_comment(comment_generation.get("comment") or "")
            for result in comment_generation.get("results") or []:
                if not force and "scores" in result:
                    continue
                prediction = normalize_comment(result.get("comment_text") or "")
                if result.get("error") or not prediction or not reference:
                    result["scores"] = None
                    continue
                pending.append((result, prediction, reference))
    return pending


def _write_records_atomically(records: list[dict], jsonl_path: Path) -> None:
    temp_path = jsonl_path.with_suffix(".jsonl.tmp")
    with temp_path.open("w", encoding="utf-8") as temp_file:
        for record in records:
            temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp_path, jsonl_path)


def _report_aggregates(records: list[dict], generation_id: str) -> None:
    """Mean of each metric per model over the generation, with the number of
    scored and unusable (scores: null) results."""
    per_model_scores = defaultdict(lambda: defaultdict(list))
    per_model_skipped = defaultdict(int)
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            for result in comment_generation.get("results") or []:
                model = result.get("model") or "<unknown>"
                scores = result.get("scores")
                if scores is None:
                    per_model_skipped[model] += 1
                    continue
                for metric in METRIC_NAMES:
                    per_model_scores[model][metric].append(scores[metric])

    print(f"\n{generation_id}")
    for model in sorted(set(per_model_scores) | set(per_model_skipped)):
        metrics = per_model_scores[model]
        counts = f"n={len(metrics.get('bleu4', []))}"
        if per_model_skipped[model]:
            counts += f", unusable={per_model_skipped[model]}"
        means = "  ".join(
            f"{metric}={sum(values) / len(values):.4f}"
            for metric, values in ((m, metrics[m]) for m in METRIC_NAMES)
            if values
        )
        print(f"  {model} ({counts})  {means or 'no scored results'}")


def evaluate_generation(gen_dir: Path, scorer: CommentScorer, force: bool) -> None:
    records = load_from_jsonl(gen_dir, GENERATED_FILENAME)
    pending = _collect_unscored(records, force)

    if pending:
        predictions = [prediction for _, prediction, _ in pending]
        references = [reference for _, _, reference in pending]
        for (result, _, _), scores in zip(
            pending, scorer.score_pairs(predictions, references)
        ):
            result["scores"] = scores

    _write_records_atomically(records, gen_dir / f"{GENERATED_FILENAME}.jsonl")
    logging.info("Scored %d new results in %s", len(pending), gen_dir)
    _report_aggregates(records, gen_dir.name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to evaluate (defaults to the latest run)",
    )
    parser.add_argument(
        "--generation",
        type=str,
        default=None,
        help="Only evaluate this generation label (defaults to all in the run)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute scores for results that already have them",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else require_latest_run_directory()
    generations = list_generations(run_dir)
    if args.generation:
        generations = [gen for gen in generations if gen["id"] == args.generation]
        if not generations:
            raise SystemExit(f"No generation '{args.generation}' in {run_dir}")
    if not generations:
        raise SystemExit(f"No generations with output found in {run_dir}")

    scorer = CommentScorer()
    for generation in generations:
        evaluate_generation(generation["dir"], scorer, force=args.force)


if __name__ == "__main__":
    main()
