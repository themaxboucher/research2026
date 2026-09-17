import argparse
import json
import logging
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from analyze.correlation import CORRELATIONS_FILENAME
from eval.constants import BLEU_METRICS
from storage.runs import resolve_dataset_and_run

SCORE_LABELS = {
    **{metric: f"BLEU-{metric.removeprefix('bleu')}" for metric in BLEU_METRICS},
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore",
}
COMPLEXITY_LABELS = {
    "cyclomatic_complexity": "Cyclomatic Complexity",
    "cognitive_complexity": "Cognitive Complexity",
}
CORRELATION_METHODS = ("pearson", "spearman")


def read_correlations_json(run_directory: Path) -> list[dict]:
    correlations_json_path = run_directory / CORRELATIONS_FILENAME
    with open(correlations_json_path, "r") as correlations_file:
        return json.load(correlations_file)


def _format_correlation(correlation: dict, method: str) -> str:
    coefficient = correlation.get(f"{method}_r")
    if coefficient is None:
        return "—"
    p_value = correlation.get(f"{method}_p_value")
    significance = "*" if p_value is not None and p_value < 0.05 else ""
    return f"{coefficient:.4f}{significance}"


def _draw_matrix(axes, correlations: dict, model: str, method: str) -> None:
    row_count = len(SCORE_LABELS)
    axes.set_xlim(0, 1)
    axes.set_ylim(-0.5, row_count + 3)
    axes.axis("off")
    axes.set_title(f"{model} — {method.title()}", fontsize=11, pad=8)
    axes.hlines([0, row_count + 2.7], 0, 1, color="black", linewidth=0.8)
    axes.hlines(row_count + 0.6, 0, 1, color="0.4", linewidth=0.6)
    axes.hlines(row_count + 1.6, 0.4, 1, color="black", linewidth=0.6)
    axes.text(0.7, row_count + 2.1, "Code Quality Metric", ha="center", va="center")
    axes.text(0.02, row_count / 2, "NLP Metrics", va="center")

    for column, label in enumerate(COMPLEXITY_LABELS.values()):
        axes.text(0.55 + column * 0.3, row_count + 1.1, label, ha="center", va="center")

    for row, label in enumerate(SCORE_LABELS.values()):
        axes.text(0.22, row_count - row - 0.2, label, va="center")

    for (row, score), (column, complexity) in product(
        enumerate(SCORE_LABELS), enumerate(COMPLEXITY_LABELS)
    ):
        correlation = correlations.get((model, score, complexity), {})
        axes.text(
            0.55 + column * 0.3,
            row_count - row - 0.2,
            _format_correlation(correlation, method),
            ha="center",
            va="center",
        )


def plot_matrices(correlations: list[dict], output_path: Path) -> None:
    if not correlations:
        logging.warning("No correlations to plot, writing no figure")
        return

    models = sorted({correlation["model"] for correlation in correlations})
    indexed_correlations = {
        (row["model"], row["score_metric"], row["complexity_metric"]): row
        for row in correlations
    }
    with plt.rc_context({"font.family": "serif", "font.size": 10}):
        figure, axes_grid = plt.subplots(
            len(models), 2, figsize=(14, 2.8 * len(models)), squeeze=False
        )
        for (row, model), (column, method) in product(
            enumerate(models), enumerate(CORRELATION_METHODS)
        ):
            _draw_matrix(axes_grid[row, column], indexed_correlations, model, method)

        figure.text(
            0.5,
            0.01,
            "* Statistically significant at the 5% significance level. "
            "— Unavailable correlation.",
            ha="center",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.5, w_pad=2)
        figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    logging.info("Wrote %s", output_path)


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
    correlations = read_correlations_json(run_directory)
    plot_matrices(correlations, run_directory / "correlations_matrix.png")


if __name__ == "__main__":
    main()
