import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless backend so figures render without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runs import require_latest_run_directory
from storage import load_from_json

GENERATED_DATASET_FILENAME = "files_generated"
ORIGINAL_SOURCE_LABEL = "original"

HISTOGRAM_BIN_COUNT = 30
SOURCE_COLOR_PALETTE = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
]


class SourceVariant(NamedTuple):
    source_label: str
    content: str
    comments: list[dict]


def _repo_identifier(source_file: dict) -> str:
    return f"{source_file['repo_owner']}/{source_file['repo_name']}"


def _short_source_label(source: str) -> str:
    return source.split("/")[-1] if "/" in source else source


def compute_content_metrics(content: str, comments: list[dict]) -> dict:
    line_count = len(content.splitlines())
    comment_count = len(comments)
    comments_per_loc = comment_count / line_count if line_count else 0.0
    total_comment_characters = sum(len(comment["comment"]) for comment in comments)
    avg_comment_length = (
        total_comment_characters / comment_count if comment_count else 0.0
    )
    return {
        "loc": line_count,
        "comments": comment_count,
        "comments_per_loc": comments_per_loc,
        "avg_comment_length": avg_comment_length,
    }


def _iter_source_variants(source_file: dict):
    original_content = source_file.get("content")
    if original_content:
        yield SourceVariant(
            ORIGINAL_SOURCE_LABEL,
            original_content,
            source_file.get("comments", []),
        )

    for generation in source_file.get("generations", []):
        generated_content = generation.get("generated_content")
        if generated_content:
            yield SourceVariant(
                generation["model"],
                generated_content,
                generation.get("generated_comments", []),
            )


def build_file_metrics_dataframe(generated_files: list[dict]) -> pd.DataFrame:
    metric_rows = []
    for source_file in generated_files:
        repo = _repo_identifier(source_file)
        for variant in _iter_source_variants(source_file):
            metrics = compute_content_metrics(variant.content, variant.comments)
            metric_rows.append(
                {
                    "repo": repo,
                    "filepath": source_file["filepath"],
                    "source": variant.source_label,
                    **metrics,
                }
            )
    return pd.DataFrame(metric_rows)


def _build_comment_row(repo: str, source_label: str, comment: dict) -> dict:
    line_span = comment["end_line"] - comment["start_line"] + 1
    return {
        "repo": repo,
        "source": source_label,
        "comment_type": comment["type"],
        "character_length": len(comment["comment"]),
        "line_span": line_span,
        "is_multiline": line_span > 1,
    }


def build_comments_dataframe(generated_files: list[dict]) -> pd.DataFrame:
    comment_rows = []
    for source_file in generated_files:
        repo = _repo_identifier(source_file)
        for variant in _iter_source_variants(source_file):
            for comment in variant.comments:
                comment_rows.append(
                    _build_comment_row(repo, variant.source_label, comment)
                )
    return pd.DataFrame(comment_rows)


def _ordered_sources(file_metrics: pd.DataFrame) -> list[str]:
    present_sources = set(file_metrics["source"].unique())
    ordered = []
    if ORIGINAL_SOURCE_LABEL in present_sources:
        ordered.append(ORIGINAL_SOURCE_LABEL)
    ordered.extend(
        sorted(source for source in present_sources if source != ORIGINAL_SOURCE_LABEL)
    )
    return ordered


def _source_colors(sources: list[str]) -> dict[str, str]:
    return {
        source: SOURCE_COLOR_PALETTE[index % len(SOURCE_COLOR_PALETTE)]
        for index, source in enumerate(sources)
    }


def compute_per_repo_summary(
    file_metrics: pd.DataFrame, comments: pd.DataFrame
) -> pd.DataFrame:
    if not comments.empty:
        type_counts_by_group = (
            comments.groupby(["repo", "source", "comment_type"])
            .size()
            .unstack(fill_value=0)
        )
    else:
        type_counts_by_group = pd.DataFrame()

    def _type_count(repo: str, source: str, comment_type: str) -> int:
        if comment_type not in type_counts_by_group.columns:
            return 0
        if (repo, source) not in type_counts_by_group.index:
            return 0
        return int(type_counts_by_group.at[(repo, source), comment_type])

    summary_rows = []
    for (repo, source), group in file_metrics.groupby(["repo", "source"]):
        summary_rows.append(
            {
                "repo": repo,
                "source": source,
                "total_files": int(len(group)),
                "total_loc": int(group["loc"].sum()),
                "total_comments": int(group["comments"].sum()),
                "inline_comments": _type_count(repo, source, "inline"),
                "block_comments": _type_count(repo, source, "block"),
                "docstring_comments": _type_count(repo, source, "docstring"),
                "files_without_comments": int((group["comments"] == 0).sum()),
                "mean_comments_per_file": float(group["comments"].mean()),
                "mean_comments_per_loc": float(group["comments_per_loc"].mean()),
                "median_comments_per_loc": float(group["comments_per_loc"].median()),
                "mean_avg_comment_length": float(group["avg_comment_length"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary.sort_values(["repo", "source"]).reset_index(drop=True)


def _save_figure(figure, reports_dir: Path, output_filename: str) -> None:
    figure.savefig(reports_dir / output_filename, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_distribution_by_source(
    values_frame: pd.DataFrame,
    value_column: str,
    sources: list[str],
    source_colors: dict[str, str],
    reports_dir: Path,
    *,
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    output_filename: str,
) -> None:
    combined_values = values_frame[value_column].dropna()
    if combined_values.empty:
        return

    bin_edges = np.histogram_bin_edges(combined_values, bins=HISTOGRAM_BIN_COUNT)
    figure, axes = plt.subplots(figsize=(9, 5))
    for source in sources:
        source_values = values_frame.loc[
            values_frame["source"] == source, value_column
        ].dropna()
        if source_values.empty:
            continue
        axes.hist(
            source_values,
            bins=bin_edges,
            alpha=0.55,
            label=_short_source_label(source),
            color=source_colors[source],
            edgecolor="white",
        )
    axes.set_title(title)
    axes.set_xlabel(x_axis_label)
    axes.set_ylabel(y_axis_label)
    axes.legend(title="Source")
    _save_figure(figure, reports_dir, output_filename)


def plot_metric_by_source(
    per_source_summary: pd.DataFrame,
    metric_column: str,
    sources: list[str],
    source_colors: dict[str, str],
    reports_dir: Path,
    *,
    title: str,
    y_axis_label: str,
    output_filename: str,
) -> None:
    labels = [_short_source_label(source) for source in sources]
    heights = [per_source_summary.loc[source, metric_column] for source in sources]
    colors = [source_colors[source] for source in sources]

    figure, axes = plt.subplots(figsize=(9, 5))
    bars = axes.bar(labels, heights, color=colors)
    axes.set_title(title)
    axes.set_ylabel(y_axis_label)
    axes.bar_label(bars, fmt="%.3g", padding=3)
    figure.autofmt_xdate(rotation=30)
    _save_figure(figure, reports_dir, output_filename)


def _render_all_plots(
    file_metrics: pd.DataFrame,
    sources: list[str],
    source_colors: dict[str, str],
    reports_dir: Path,
) -> None:
    plot_distribution_by_source(
        file_metrics,
        "comments_per_loc",
        sources,
        source_colors,
        reports_dir,
        title="Comment density: original vs. LLMs",
        x_axis_label="Comments per line of code",
        y_axis_label="Number of files",
        output_filename="comments_per_loc_by_source.png",
    )
    plot_distribution_by_source(
        file_metrics,
        "comments",
        sources,
        source_colors,
        reports_dir,
        title="Comments per file: original vs. LLMs",
        x_axis_label="Comment count",
        y_axis_label="Number of files",
        output_filename="comments_per_file_by_source.png",
    )


def _write_summary_csv(summary: pd.DataFrame, reports_dir: Path) -> None:
    summary.to_csv(reports_dir / "summary.csv", index=False)


def generate_report(run_dir: Path) -> None:
    generated_files = load_from_json(run_dir, GENERATED_DATASET_FILENAME)
    file_metrics = build_file_metrics_dataframe(generated_files)

    if file_metrics.empty:
        logging.warning("No files with content found; nothing to report.")
        return

    comments = build_comments_dataframe(generated_files)
    sources = _ordered_sources(file_metrics)
    source_colors = _source_colors(sources)

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_per_repo_summary(file_metrics, comments)

    _write_summary_csv(summary, reports_dir)
    _render_all_plots(file_metrics, sources, source_colors, reports_dir)

    logging.info(
        "Report written to %s (sources compared: %s)",
        reports_dir,
        ", ".join(sources),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    generate_report(require_latest_run_directory())
