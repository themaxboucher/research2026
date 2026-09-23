import argparse
import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from itertools import product
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from eval.constants import SCORE_METRICS, UNKNOWN_MODEL
from generate.constants import GENERATE_FILENAME
from storage import iter_from_jsonl
from storage.runs import resolve_dataset_and_run

COMMENT_GENERATION_METRICS = (
    "cyclomatic_complexity",
    "cognitive_complexity",
    "lines_of_code",
    "logical_lines_of_code",
    "comment_density",
    "reference_comment_length",
)
RESULT_METRICS = ("generated_comment_length",)
COMPLEXITY_METRICS = COMMENT_GENERATION_METRICS + RESULT_METRICS
CORRELATIONS_FILENAME = "correlations.json"


def _scored_results(comment_generation: dict):
    for result in comment_generation.get("results") or []:
        if result.get("error") or not result.get("scores"):
            continue
        yield result.get("model") or UNKNOWN_MODEL, result


def _iter_samples(records: Iterable[dict]):
    for record in records:
        yield from record.get("comment_generations") or []


def _samples_by_model(records: Iterable[dict]) -> dict[str, list[dict]]:
    samples_by_model = defaultdict(list)
    for comment_generation in _iter_samples(records):
        comment_generation_metrics = {
            metric: comment_generation.get(metric)
            for metric in COMMENT_GENERATION_METRICS
        }
        for model, result in _scored_results(comment_generation):
            scores = result["scores"]
            samples_by_model[model].append(
                {
                    **{metric: scores.get(metric) for metric in SCORE_METRICS},
                    **comment_generation_metrics,
                    **{metric: result.get(metric) for metric in RESULT_METRICS},
                }
            )
    return samples_by_model


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _correlate_pair(
    samples: list[dict], score_metric: str, complexity_metric: str
) -> dict:
    pairs = [
        (sample[score_metric], sample[complexity_metric])
        for sample in samples
        if _is_finite_number(sample.get(score_metric))
        and _is_finite_number(sample.get(complexity_metric))
    ]
    correlation = {
        "score_metric": score_metric,
        "complexity_metric": complexity_metric,
        "n": len(pairs),
        "pearson_r": None,
        "pearson_p_value": None,
        "spearman_r": None,
        "spearman_p_value": None,
        "status": "insufficient_data",
    }
    if len(pairs) < 2:
        return correlation

    score_values, complexity_values = zip(*pairs)
    if len(set(score_values)) == 1 or len(set(complexity_values)) == 1:
        correlation["status"] = "constant_input"
        return correlation

    pearson = pearsonr(score_values, complexity_values, alternative="two-sided")
    spearman = spearmanr(score_values, complexity_values, alternative="two-sided")
    correlation.update(
        pearson_r=_finite_or_none(pearson.statistic),
        pearson_p_value=_finite_or_none(pearson.pvalue),
        spearman_r=_finite_or_none(spearman.statistic),
        spearman_p_value=_finite_or_none(spearman.pvalue),
        status="ok" if len(pairs) > 2 else "spearman_p_value_unavailable",
    )
    return correlation


def compute_correlations(records: Iterable[dict]) -> list[dict]:
    samples_by_model = _samples_by_model(records)
    return [
        {
            "model": model,
            **_correlate_pair(samples_by_model[model], score_metric, complexity_metric),
        }
        for model, score_metric, complexity_metric in product(
            sorted(samples_by_model), SCORE_METRICS, COMPLEXITY_METRICS
        )
    ]


def _correlate_run(run_dir: Path) -> None:
    records = iter_from_jsonl(run_dir, GENERATE_FILENAME + "_scored")
    correlations = compute_correlations(records)
    if not correlations:
        logging.warning("No scored results found in %s", run_dir)
    elif not any(row["n"] >= 2 for row in correlations):
        logging.warning(
            "Too few paired samples; run analyze.analyze to add code and comment metrics"
        )
    output_path = run_dir / CORRELATIONS_FILENAME
    output_path.write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logging.info("Wrote %d correlations to %s", len(correlations), output_path)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Dataset directory to analyze (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory to analyze (defaults to the latest run in the dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    _, run_directory = resolve_dataset_and_run(args.dataset_dir, args.run_dir)
    _correlate_run(run_directory)


if __name__ == "__main__":
    main()
