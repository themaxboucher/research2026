import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import evaluate
from tqdm.auto import tqdm

from generate import LOCATION_FILENAME, REGENERATE_FILENAME, list_generations
from runs import require_latest_run_directory
from storage import load_from_jsonl


def normalize_comment(text: str) -> str:
    """Comment text reduced to its content. Per-line '#' markers and indentation 
    are stripped. Evrything is joined into a single line with single spaces."""
    words = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        words.extend(stripped.split())
    return " ".join(words)


class CommentScorer:
    """Computes the three metrics over batches of (prediction, reference) pairs."""

    def __init__(self):
        self._bleu = None
        self._rouge = None
        self._bertscore = None

    def _load(self):
        # Load the metric models if they haven't been loaded yet.
        # We load them lazily so that runs with no results to score don't pay the cost of loading BERTScore.
        if self._rouge is None or self._bertscore is None:
            self._rouge = evaluate.load("rouge")
            self._bertscore = evaluate.load("bertscore")

    def _load_bleu(self):
        # BLEU loads separately because corpus_bleu is needed at aggregate
        # time even when there are no new pairs to score.
        if self._bleu is None:
            self._bleu = evaluate.load("bleu")

    def corpus_bleu(self, predictions: list[str], references: list[str]) -> float:
        """Standard (unsmoothed) corpus-level BLEU-4 over all pairs at once."""
        self._load_bleu()
        return self._bleu.compute(
            predictions=predictions, references=references, max_order=4
        )["bleu"]

    def score_pairs(
        self,
        predictions: list[str],
        references: list[str],
        desc: str = "Scoring",
    ) -> list[dict]:
        self._load()
        self._load_bleu()

        # Score in chunks so a progress bar can advance as pairs are processed,
        # rather than blocking on one opaque batch (BERTScore dominates the cost).
        scores = []
        batch_size = 64
        with tqdm(total=len(predictions), desc=desc, unit="pair") as progress_bar:
            for start in range(0, len(predictions), batch_size):
                pred_batch = predictions[start : start + batch_size]
                ref_batch = references[start : start + batch_size]

                # We use per comment pair BLEU 4 here (not corpus-level).
                # Smoothing is neeeded, otherwise any pair without a matching
                # 4-gram scores 0.
                bleu_scores = [
                    self._bleu.compute(
                        predictions=[pred], references=[ref], max_order=4, smooth=True
                    )["bleu"]
                    for pred, ref in zip(pred_batch, ref_batch)
                ]
                rouge_scores = self._rouge.compute(
                    predictions=pred_batch,
                    references=ref_batch,
                    rouge_types=["rougeL"],
                    use_aggregator=False,
                )["rougeL"]
                bertscore_f1 = self._bertscore.compute(
                    predictions=pred_batch, references=ref_batch, lang="en"
                )["f1"]

                scores.extend(
                    {"bleu4": bleu, "rougeL": rouge, "bertscore_f1": bert}
                    for bleu, rouge, bert in zip(
                        bleu_scores, rouge_scores, bertscore_f1
                    )
                )
                progress_bar.update(len(pred_batch))

        return scores


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


def _write_metrics(records: list[dict], gen_dir: Path, scorer: CommentScorer) -> None:
    """Per-model aggregates saved to metrics.json in the generation directory.

    BLEU is corpus-level, so it is recomputed from the normalized texts of all
    usable results rather than averaged from the per-pair sentence scores.
    ROUGE-L and BERTScore average cleanly per pair, so their stored scores are
    reused. Models with no usable results are omitted."""
    per_model_pairs = defaultdict(list)
    per_model_scores = defaultdict(lambda: defaultdict(list))
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            reference = normalize_comment(comment_generation.get("comment") or "")
            for result in comment_generation.get("results") or []:
                if result.get("scores") is None:
                    continue
                model = result.get("model") or "<unknown>"
                prediction = normalize_comment(result.get("comment_text") or "")
                per_model_pairs[model].append((prediction, reference))
                for metric in ("rougeL", "bertscore_f1"):
                    per_model_scores[model][metric].append(result["scores"][metric])

    metrics = {}
    for model in sorted(per_model_pairs):
        predictions = [prediction for prediction, _ in per_model_pairs[model]]
        references = [reference for _, reference in per_model_pairs[model]]
        scores = per_model_scores[model]
        metrics[model] = {
            "bleu4_corpus": scorer.corpus_bleu(predictions, references),
            "rougeL": sum(scores["rougeL"]) / len(scores["rougeL"]),
            "bertscore_f1": sum(scores["bertscore_f1"]) / len(scores["bertscore_f1"]),
        }

    metrics_path = gen_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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


def _collect_unscored_regenerations(
    records: list[dict], force: bool
) -> list[tuple[dict, str, str]]:
    """Placement hits that still need scoring, as (extraction, prediction,
    reference) with prediction/reference already normalized. Misses and
    unusable extractions get `scores: null` right away and are not returned."""
    pending = []
    for _, target, extraction in _iter_valid_extractions(records):
        if not force and "scores" in extraction:
            continue
        reference = normalize_comment(target.get("comment") or "")
        prediction = normalize_comment(extraction.get("comment_text") or "")
        if not extraction.get("placement_hit") or not prediction or not reference:
            extraction["scores"] = None
            continue
        pending.append((extraction, prediction, reference))
    return pending


def _write_regeneration_metrics(
    records: list[dict], gen_dir: Path, scorer: CommentScorer
) -> None:
    """Per-model aggregates saved to metrics_regenerate.json in the generation
    directory. Alongside the text metrics (computed over placement hits only),
    each model gets: regen_failure_rate — scope regenerations rejected because
    the code changed or the output didn't parse; placement_recall — targets
    the model commented at, out of all targets in valid regenerations;
    form_match_rate — hits whose comment form (inline/block) matched the
    human's."""
    scope_counts = defaultdict(lambda: {"total": 0, "failed": 0})
    for record in records:
        for result in record.get("results") or []:
            model = result.get("model") or "<unknown>"
            scope_counts[model]["total"] += 1
            if result.get("error"):
                scope_counts[model]["failed"] += 1

    target_counts = defaultdict(lambda: {"targets": 0, "hits": 0, "form_matches": 0})
    per_model_pairs = defaultdict(list)
    per_model_scores = defaultdict(lambda: defaultdict(list))
    for result, target, extraction in _iter_valid_extractions(records):
        model = result.get("model") or "<unknown>"
        counts = target_counts[model]
        counts["targets"] += 1
        if extraction.get("placement_hit"):
            counts["hits"] += 1
            if extraction.get("form_matches"):
                counts["form_matches"] += 1
        if extraction.get("scores") is not None:
            reference = normalize_comment(target.get("comment") or "")
            prediction = normalize_comment(extraction.get("comment_text") or "")
            per_model_pairs[model].append((prediction, reference))
            for metric in ("rougeL", "bertscore_f1"):
                per_model_scores[model][metric].append(extraction["scores"][metric])

    metrics = {}
    for model in sorted(scope_counts):
        scopes = scope_counts[model]
        counts = target_counts[model]
        model_metrics = {
            "scope_count": scopes["total"],
            "regen_failure_rate": scopes["failed"] / scopes["total"],
            "placement_recall": (
                counts["hits"] / counts["targets"] if counts["targets"] else None
            ),
            "form_match_rate": (
                counts["form_matches"] / counts["hits"] if counts["hits"] else None
            ),
        }
        pairs = per_model_pairs[model]
        if pairs:
            predictions = [prediction for prediction, _ in pairs]
            references = [reference for _, reference in pairs]
            scores = per_model_scores[model]
            model_metrics["bleu4_corpus"] = scorer.corpus_bleu(predictions, references)
            model_metrics["rougeL"] = sum(scores["rougeL"]) / len(scores["rougeL"])
            model_metrics["bertscore_f1"] = sum(scores["bertscore_f1"]) / len(
                scores["bertscore_f1"]
            )
        metrics[model] = model_metrics

    metrics_path = gen_dir / "metrics_regenerate.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _evaluate_location_generation(
    gen_dir: Path, scorer: CommentScorer, force: bool
) -> None:
    records = load_from_jsonl(gen_dir, LOCATION_FILENAME)
    pending = _collect_unscored(records, force)

    if pending:
        predictions = [prediction for _, prediction, _ in pending]
        references = [reference for _, _, reference in pending]
        desc = f"Scoring {gen_dir.name}"
        for (result, _, _), scores in zip(
            pending, scorer.score_pairs(predictions, references, desc=desc)
        ):
            result["scores"] = scores

    _write_records_atomically(records, gen_dir / f"{LOCATION_FILENAME}.jsonl")
    logging.info("Scored %d new results in %s", len(pending), gen_dir)
    _write_metrics(records, gen_dir, scorer)


def _evaluate_regeneration(gen_dir: Path, scorer: CommentScorer, force: bool) -> None:
    records = load_from_jsonl(gen_dir, REGENERATE_FILENAME)
    pending = _collect_unscored_regenerations(records, force)

    if pending:
        predictions = [prediction for _, prediction, _ in pending]
        references = [reference for _, _, reference in pending]
        desc = f"Scoring {gen_dir.name} (regenerate)"
        for (extraction, _, _), scores in zip(
            pending, scorer.score_pairs(predictions, references, desc=desc)
        ):
            extraction["scores"] = scores

    _write_records_atomically(records, gen_dir / f"{REGENERATE_FILENAME}.jsonl")
    logging.info("Scored %d new regeneration results in %s", len(pending), gen_dir)
    _write_regeneration_metrics(records, gen_dir, scorer)


def evaluate_generation(gen_dir: Path, scorer: CommentScorer, force: bool) -> None:
    if (gen_dir / f"{LOCATION_FILENAME}.jsonl").exists():
        _evaluate_location_generation(gen_dir, scorer, force)
    if (gen_dir / f"{REGENERATE_FILENAME}.jsonl").exists():
        _evaluate_regeneration(gen_dir, scorer, force)


def parse_args():
    parser = argparse.ArgumentParser()
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
    for generation in tqdm(generations, desc="Generations", unit="gen"):
        evaluate_generation(generation["dir"], scorer, force=args.force)


if __name__ == "__main__":
    main()
