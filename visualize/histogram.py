import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from eval.constants import (
    BASELINE_MODELS,
    BLEU_METRICS,
    SCORE_METRICS,
    UNKNOWN_MODEL,
)
from generate.constants import GENERATE_FILENAME
from storage import load_from_jsonl
from storage.runs import resolve_dataset_and_run

METRIC_LABELS = {
    **{metric: f"BLEU-{metric.removeprefix('bleu')}" for metric in BLEU_METRICS},
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
}

BIN_COUNT = 50
FILL_ALPHA = 0.25
OUTLINE_WIDTH = 1.8
BASELINE_ALPHA = 0.6
BASELINE_WIDTH = 1.2
BASELINE_STYLE = ":"
LEGEND_MAX_COLUMNS = 3
LEGEND_ROW_HEIGHT = 0.07
# One panel per score metric, so the figure widens with the metric count rather
# than squeezing every added BLEU order into a fixed width.
PANEL_WIDTH = 5
FIGURE_SIZE = (PANEL_WIDTH * len(SCORE_METRICS), 5)
FIGURE_DPI = 150
PALETTE = "tab10"


def _iter_scores(records: list[dict]):
    for record in records:
        for comment_generation in record.get("comment_generations") or []:
            for result in comment_generation.get("results") or []:
                scores = result.get("scores")
                if scores is not None:
                    yield result.get("model") or UNKNOWN_MODEL, scores


def _values_by_model(score_pairs) -> dict[str, dict[str, list[float]]]:
    values_by_model = defaultdict(lambda: defaultdict(list))
    for model, scores in score_pairs:
        for metric in SCORE_METRICS:
            value = scores.get(metric)
            if value is not None:
                values_by_model[model][metric].append(value)
    return values_by_model


def _model_colors(models: list[str]) -> dict[str, tuple]:
    palette = plt.get_cmap(PALETTE)
    return {model: palette(index % palette.N) for index, model in enumerate(models)}


def _plot_metric(axes, metric: str, values_by_model: dict, colors: dict) -> None:
    plotted = {
        model: metrics[metric]
        for model, metrics in values_by_model.items()
        if metrics[metric]
    }
    axes.set_title(METRIC_LABELS.get(metric, metric))
    axes.set_xlabel("score")
    axes.set_ylabel("density")
    if not plotted:
        return

    # Every model shares one bin count and one range, so the overlaid bars line
    # up bin for bin and the curves are actually comparable.
    lowest = min(min(values) for values in plotted.values())
    highest = max(max(values) for values in plotted.values())
    for model in sorted(plotted):
        values = plotted[model]
        shared_options = {
            "bins": BIN_COUNT,
            "range": (lowest, highest),
            "density": True,
            "color": colors[model],
        }
        label = f"{model} (n={len(values)})"
        if model in BASELINE_MODELS:
            axes.hist(
                values,
                histtype="step",
                label=label,
                linestyle=BASELINE_STYLE,
                linewidth=BASELINE_WIDTH,
                alpha=BASELINE_ALPHA,
                **shared_options,
            )
            continue
        axes.hist(
            values,
            histtype="stepfilled",
            alpha=FILL_ALPHA,
            label=label,
            **shared_options,
        )
        # The outline is drawn separately so it stays fully opaque over the
        # translucent fills. It carries no label, to keep the legend to one
        # entry per model.
        axes.hist(values, histtype="step", linewidth=OUTLINE_WIDTH, **shared_options)


def _plot_scores(values_by_model: dict, title: str, output_path: Path) -> None:
    models = sorted(values_by_model)
    colors = _model_colors(models)

    figure, axes_row = plt.subplots(1, len(SCORE_METRICS), figsize=FIGURE_SIZE)
    for axes, metric in zip(axes_row, SCORE_METRICS):
        _plot_metric(axes, metric, values_by_model, colors)

    figure.suptitle(title)
    handles, labels = axes_row[0].get_legend_handles_labels()
    # Three baselines on top of the models overflow a single legend row, so wrap
    # once the entries stop fitting across the figure.
    legend_rows = -(-len(models) // LEGEND_MAX_COLUMNS)
    figure.legend(
        handles, labels, loc="lower center", ncols=min(len(models), LEGEND_MAX_COLUMNS)
    )
    # Leave room at the bottom for the shared legend and at the top for the title.
    figure.tight_layout(rect=(0, LEGEND_ROW_HEIGHT * legend_rows, 1, 0.96))
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def _plot(run_dir: Path, dataset_dir: Path, filename: str) -> None:
    scored_filename = filename + "_scored"
    if not (run_dir / f"{scored_filename}.jsonl").exists():
        logging.info("No %s.jsonl in %s, skipping", scored_filename, run_dir.name)
        return

    records = load_from_jsonl(run_dir, scored_filename)
    values_by_model = _values_by_model(_iter_scores(records))
    if not values_by_model:
        logging.warning(
            "No scored results in %s.jsonl, writing no figure", scored_filename
        )
        return

    output_path = run_dir / f"{filename.split('_')[0]}_scores_histogram.png"
    _plot_scores(
        values_by_model,
        f"{filename} scores — {dataset_dir.name} / {run_dir.name}",
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

    _plot(run_directory, dataset_directory, GENERATE_FILENAME)


if __name__ == "__main__":
    main()
