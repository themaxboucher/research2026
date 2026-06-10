import logging
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")  # headless backend so figures render without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from storage import iter_from_jsonl

ORIGINAL_SOURCE_LABEL = "original"
ALL_REPOS_LABEL = "ALL"

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


def _source_file_path(source_file: dict) -> str:
    return source_file.get("new_path") or source_file["filename"]


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
    original_content = source_file.get("source_code")
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


def _present_comments(comments: list[dict]) -> list[dict]:
    return [comment for comment in comments if comment.get("status") != "removed"]


def _count_comment_types(
    repo: str, source_label: str, comments: list[dict]
) -> Counter:
    type_counts: Counter = Counter()
    for comment in comments:
        type_counts[(repo, source_label, comment["type"])] += 1
    return type_counts


def collect_metrics(source_files) -> tuple[pd.DataFrame, Counter]:
    metric_rows = []
    comment_type_counts: Counter = Counter()

    for source_file in source_files:
        repo = source_file["repo_name"]
        filepath = _source_file_path(source_file)
        for variant in _iter_source_variants(source_file):
            present_comments = _present_comments(variant.comments)
            metrics = compute_content_metrics(variant.content, present_comments)
            metric_rows.append(
                {
                    "repo": repo,
                    "filepath": filepath,
                    "source": variant.source_label,
                    **metrics,
                }
            )
            comment_type_counts.update(
                _count_comment_types(repo, variant.source_label, present_comments)
            )

    return pd.DataFrame(metric_rows), comment_type_counts


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


def _summary_row(
    repo: str,
    source: str,
    files: pd.DataFrame,
    inline_count: int,
    block_count: int,
    docstring_count: int,
) -> dict:
    return {
        "repo": repo,
        "source": source,
        "total_files": int(len(files)),
        "total_loc": int(files["loc"].sum()),
        "total_comments": int(files["comments"].sum()),
        "inline_comments": inline_count,
        "block_comments": block_count,
        "docstring_comments": docstring_count,
        "files_without_comments": int((files["comments"] == 0).sum()),
        "mean_comments_per_file": float(files["comments"].mean()),
        "mean_comments_per_loc": float(files["comments_per_loc"].mean()),
        "median_comments_per_loc": float(files["comments_per_loc"].median()),
        "mean_avg_comment_length": float(files["avg_comment_length"].mean()),
    }


def _per_repo_summary_rows(
    file_metrics: pd.DataFrame, comment_type_counts: Counter
) -> list[dict]:
    rows = []
    for (repo, source), files in file_metrics.groupby(["repo", "source"]):
        rows.append(
            _summary_row(
                repo,
                source,
                files,
                comment_type_counts.get((repo, source, "inline"), 0),
                comment_type_counts.get((repo, source, "block"), 0),
                comment_type_counts.get((repo, source, "docstring"), 0),
            )
        )
    return rows


def _total_type_count(
    comment_type_counts: Counter, source: str, comment_type: str
) -> int:
    return sum(
        count
        for (_, counted_source, counted_type), count in comment_type_counts.items()
        if counted_source == source and counted_type == comment_type
    )


def _all_repos_summary_rows(
    file_metrics: pd.DataFrame, comment_type_counts: Counter
) -> list[dict]:
    rows = []
    for source in _ordered_sources(file_metrics):
        files = file_metrics[file_metrics["source"] == source]
        rows.append(
            _summary_row(
                ALL_REPOS_LABEL,
                source,
                files,
                _total_type_count(comment_type_counts, source, "inline"),
                _total_type_count(comment_type_counts, source, "block"),
                _total_type_count(comment_type_counts, source, "docstring"),
            )
        )
    return rows


def compute_per_repo_summary(
    file_metrics: pd.DataFrame, comment_type_counts: Counter
) -> pd.DataFrame:
    all_repos_rows = _all_repos_summary_rows(file_metrics, comment_type_counts)
    per_repo_rows = _per_repo_summary_rows(file_metrics, comment_type_counts)
    return pd.DataFrame(all_repos_rows + per_repo_rows)


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


def generate_report(run_dir: Path, dataset_filename: str) -> None:
    records = iter_from_jsonl(run_dir, dataset_filename)
    file_metrics, comment_type_counts = collect_metrics(records)

    if file_metrics.empty:
        logging.warning("No files with content found; nothing to report.")
        return

    sources = _ordered_sources(file_metrics)
    source_colors = _source_colors(sources)

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_per_repo_summary(file_metrics, comment_type_counts)

    _write_summary_csv(summary, reports_dir)
    _render_all_plots(file_metrics, sources, source_colors, reports_dir)

    logging.info(
        "Report written to %s (sources compared: %s)",
        reports_dir,
        ", ".join(sources),
    )
