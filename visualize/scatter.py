import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from eval.constants import BLEU_METRICS, SCORE_METRICS, UNKNOWN_MODEL
from generate.constants import GENERATE_FILENAME
from storage import iter_from_jsonl
from storage.runs import resolve_dataset_and_run

SCORE_LABELS = {
    **{metric: f"BLEU-{metric.removeprefix('bleu')}" for metric in BLEU_METRICS},
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
}

COMPLEXITY_LABELS = {
    "cognitive_complexity": "cognitive complexity",
    "cyclomatic_complexity": "cyclomatic complexity",
}
COMPLEXITY_METRICS = tuple(COMPLEXITY_LABELS)

POINT_SIZE = 10
POINT_ALPHA = 0.3
LEGEND_MAX_COLUMNS = 3
LEGEND_ROW_HEIGHT = 0.05
# The grid is one column per score metric and one row per complexity metric, so
# the figure grows with both rather than squeezing the added BLEU orders.
PANEL_WIDTH = 5
PANEL_HEIGHT = 4.5
FIGURE_SIZE = (
    PANEL_WIDTH * len(SCORE_METRICS),
    PANEL_HEIGHT * len(COMPLEXITY_METRICS),
)
FIGURE_DPI = 150
PALETTE = "tab10"


def _iter_scored_points(records):
    """Yield (model, complexities, scores) for every fully measured generation.

    A generation whose code scope failed to parse has no complexity, and a
    result that errored has no scores; both are dropped so each panel plots the
    same set of points.
    """
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            complexities = {
                metric: comment_generation.get(metric) for metric in COMPLEXITY_METRICS
            }
            if any(value is None for value in complexities.values()):
                continue

            for result in comment_generation.get("results") or []:
                scores = result.get("scores") or {}
                if any(scores.get(metric) is None for metric in SCORE_METRICS):
                    continue
                yield result.get("model") or UNKNOWN_MODEL, complexities, scores


def _points_by_model(scored_points) -> dict[str, list[tuple[dict, dict]]]:
    points_by_model = defaultdict(list)
    for model, complexities, scores in scored_points:
        points_by_model[model].append((complexities, scores))
    return points_by_model


def _model_colors(models: list[str]) -> dict[str, tuple]:
    palette = plt.get_cmap(PALETTE)
    return {model: palette(index % palette.N) for index, model in enumerate(models)}


def _plot_metric_pair(
    axes,
    complexity_metric: str,
    score_metric: str,
    points_by_model: dict,
    colors: dict,
) -> None:
    axes.set_xlabel(COMPLEXITY_LABELS[complexity_metric])
    axes.set_ylabel(SCORE_LABELS.get(score_metric, score_metric))
    for model in sorted(points_by_model):
        points = points_by_model[model]
        complexity_values = [
            complexities[complexity_metric] for complexities, _ in points
        ]
        score_values = [scores[score_metric] for _, scores in points]
        axes.scatter(
            complexity_values,
            score_values,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=colors[model],
            edgecolors="none",
            label=f"{model} (n={len(points)})",
        )


def _plot_scores_against_complexity(
    points_by_model: dict, title: str, output_path: Path
) -> None:
    models = sorted(points_by_model)
    colors = _model_colors(models)

    figure, axes_grid = plt.subplots(
        len(COMPLEXITY_METRICS), len(SCORE_METRICS), figsize=FIGURE_SIZE
    )
    for axes_row, complexity_metric in zip(axes_grid, COMPLEXITY_METRICS):
        for axes, score_metric in zip(axes_row, SCORE_METRICS):
            _plot_metric_pair(
                axes, complexity_metric, score_metric, points_by_model, colors
            )

    figure.suptitle(title)
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    legend_rows = -(-len(models) // LEGEND_MAX_COLUMNS)
    figure.legend(
        handles, labels, loc="lower center", ncols=min(len(models), LEGEND_MAX_COLUMNS)
    )
    # Leave room at the bottom for the shared legend and at the top for the title.
    figure.tight_layout(rect=(0, LEGEND_ROW_HEIGHT * legend_rows, 1, 0.96))
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def _plot_run(run_dir: Path, dataset_dir: Path) -> None:
    filename = GENERATE_FILENAME + "_scored"
    if not (run_dir / f"{filename}.jsonl").exists():
        logging.info("No %s.jsonl in %s, skipping", filename, run_dir.name)
        return

    # The scored file runs to several gigabytes, so it is streamed a record at
    # a time rather than loaded whole.
    records = iter_from_jsonl(run_dir, filename)
    points_by_model = _points_by_model(_iter_scored_points(records))
    if not points_by_model:
        logging.warning(
            "No results with both scores and complexity in %s.jsonl, writing no figure",
            filename,
        )
        return

    output_path = run_dir / f"{GENERATE_FILENAME.split('_')[0]}_complexity_scatter.png"
    _plot_scores_against_complexity(
        points_by_model,
        f"scores vs. complexity — {dataset_dir.name} / {run_dir.name}",
        output_path,
    )
    logging.info("Wrote %s", output_path)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset directory to plot for (defaults to the latest dataset)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to plot (defaults to the latest run in the dataset)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    dataset_directory, run_directory = resolve_dataset_and_run(
        args.dataset_dir, args.run_dir
    )

    _plot_run(run_directory, dataset_directory)


if __name__ == "__main__":
    main()
