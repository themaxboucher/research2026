import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so figures render without a display

import matplotlib.pyplot as plt
import pandas as pd

from storage import load_from_json

ANALYSED_DATASET_FILENAME = "files_analysed"
REPORTS_DIRECTORY = Path(__file__).parent / "reports"

TOP_REPO_DISPLAY_COUNT = 20
HISTOGRAM_BIN_COUNT = 30
FIGURE_BAR_COLOR = "#4c72b0"


def _repo_identifier(source_file: dict) -> str:
    return f"{source_file['repo_owner']}/{source_file['repo_name']}"


def build_file_metrics_dataframe(analysed_files: list[dict]) -> pd.DataFrame:
    metric_rows = []
    for source_file in analysed_files:
        original_metrics = source_file.get("original_metrics")
        if not original_metrics:
            continue
        metric_rows.append(
            {
                "repo": _repo_identifier(source_file),
                "filepath": source_file["filepath"],
                "loc": original_metrics["loc"],
                "comments": original_metrics["comments"],
                "comments_per_loc": original_metrics["comments_per_loc"],
                "avg_comment_length": original_metrics["avg_comment_length"],
            }
        )
    return pd.DataFrame(metric_rows)


def build_comments_dataframe(analysed_files: list[dict]) -> pd.DataFrame:
    comment_rows = []
    for source_file in analysed_files:
        repo = _repo_identifier(source_file)
        for comment in source_file.get("comments", []):
            line_span = comment["end_line"] - comment["start_line"] + 1
            comment_rows.append(
                {
                    "repo": repo,
                    "character_length": len(comment["comment"]),
                    "line_span": line_span,
                    "is_multiline": line_span > 1,
                }
            )
    return pd.DataFrame(comment_rows)


def compute_dataset_summary(
    file_metrics: pd.DataFrame, comments: pd.DataFrame
) -> dict:
    multiline_comment_ratio = (
        float(comments["is_multiline"].mean()) if not comments.empty else 0.0
    )
    return {
        "total_repos": int(file_metrics["repo"].nunique()),
        "total_files": int(len(file_metrics)),
        "total_loc": int(file_metrics["loc"].sum()),
        "total_comments": int(file_metrics["comments"].sum()),
        "files_without_comments": int((file_metrics["comments"] == 0).sum()),
        "mean_comments_per_loc": float(file_metrics["comments_per_loc"].mean()),
        "median_comments_per_loc": float(file_metrics["comments_per_loc"].median()),
        "mean_avg_comment_length": float(file_metrics["avg_comment_length"].mean()),
        "multiline_comment_ratio": multiline_comment_ratio,
    }


def compute_per_repo_summary(file_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped_by_repo = file_metrics.groupby("repo")
    per_repo_summary = grouped_by_repo.agg(
        file_count=("filepath", "count"),
        total_loc=("loc", "sum"),
        total_comments=("comments", "sum"),
        mean_comments_per_loc=("comments_per_loc", "mean"),
        mean_avg_comment_length=("avg_comment_length", "mean"),
    )
    return per_repo_summary.sort_values("file_count", ascending=False)


def _save_figure(figure, output_filename: str) -> None:
    figure.savefig(REPORTS_DIRECTORY / output_filename, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_histogram(
    values: pd.Series,
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    output_filename: str,
) -> None:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.hist(values, bins=HISTOGRAM_BIN_COUNT, color=FIGURE_BAR_COLOR, edgecolor="white")
    axes.set_title(title)
    axes.set_xlabel(x_axis_label)
    axes.set_ylabel(y_axis_label)
    _save_figure(figure, output_filename)


def plot_loc_vs_comments(file_metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.scatter(
        file_metrics["loc"],
        file_metrics["comments"],
        alpha=0.5,
        color=FIGURE_BAR_COLOR,
    )
    axes.set_title("Lines of code vs. comment count")
    axes.set_xlabel("Lines of code")
    axes.set_ylabel("Comment count")
    _save_figure(figure, "loc_vs_comments_scatter.png")


def plot_top_repos_by_file_count(per_repo_summary: pd.DataFrame) -> None:
    top_repos = per_repo_summary.head(TOP_REPO_DISPLAY_COUNT).iloc[::-1]
    figure, axes = plt.subplots(figsize=(10, 6))
    axes.barh(top_repos.index, top_repos["file_count"], color=FIGURE_BAR_COLOR)
    axes.set_title(f"Top {TOP_REPO_DISPLAY_COUNT} repositories by file count")
    axes.set_xlabel("Number of files")
    _save_figure(figure, "top_repos_by_file_count.png")


def _render_all_plots(
    file_metrics: pd.DataFrame,
    comments: pd.DataFrame,
    per_repo_summary: pd.DataFrame,
) -> None:
    plot_histogram(
        file_metrics["loc"],
        "Distribution of file sizes",
        "Lines of code",
        "Number of files",
        "loc_distribution.png",
    )
    plot_histogram(
        file_metrics["comments_per_loc"],
        "Distribution of comment density",
        "Comments per line of code",
        "Number of files",
        "comments_per_loc_distribution.png",
    )
    plot_histogram(
        file_metrics["avg_comment_length"],
        "Distribution of average comment length",
        "Average comment length (characters)",
        "Number of files",
        "avg_comment_length_distribution.png",
    )
    plot_histogram(
        per_repo_summary["mean_comments_per_loc"],
        "Comment density across repositories",
        "Mean comments per line of code",
        "Number of repositories",
        "repo_comment_density_distribution.png",
    )
    plot_loc_vs_comments(file_metrics)
    plot_top_repos_by_file_count(per_repo_summary)

    if not comments.empty:
        plot_histogram(
            comments["character_length"],
            "Distribution of comment lengths",
            "Comment length (characters)",
            "Number of comments",
            "comment_length_distribution.png",
        )


def _write_summary_json(dataset_summary: dict) -> None:
    summary_path = REPORTS_DIRECTORY / "summary.json"
    summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")


def _write_per_repo_csv(per_repo_summary: pd.DataFrame) -> None:
    per_repo_summary.to_csv(REPORTS_DIRECTORY / "per_repo_metrics.csv")


def _log_dataset_summary(dataset_summary: dict) -> None:
    logging.info("Dataset summary:")
    for metric_name, metric_value in dataset_summary.items():
        logging.info("  %s: %s", metric_name, metric_value)


def generate_report() -> None:
    analysed_files = load_from_json(ANALYSED_DATASET_FILENAME)
    file_metrics = build_file_metrics_dataframe(analysed_files)

    if file_metrics.empty:
        logging.warning("No analysed files with metrics found; nothing to report.")
        return

    comments = build_comments_dataframe(analysed_files)
    REPORTS_DIRECTORY.mkdir(exist_ok=True)

    dataset_summary = compute_dataset_summary(file_metrics, comments)
    per_repo_summary = compute_per_repo_summary(file_metrics)

    _write_summary_json(dataset_summary)
    _write_per_repo_csv(per_repo_summary)
    _render_all_plots(file_metrics, comments, per_repo_summary)

    _log_dataset_summary(dataset_summary)
    logging.info("Report written to %s", REPORTS_DIRECTORY)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    generate_report()
