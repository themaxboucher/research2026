from collect import DEFAULT_MINING_WORKERS, collect_dataset
from generate import generate_comments_for_dataset
from report import generate_report
from runs import resolve_run_directory
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect", action="store_true", help="Collect data from GitHub"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate comments for collected data"
    )
    parser.add_argument(
        "--report", action="store_true", help="Build dataset statistics and graphs"
    )
    parser.add_argument(
        "--report-dataset",
        choices=["files_generated", "repo_files"],
        default="files_generated",
        help="Which dataset the report reads (default: the post-generation dataset)",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a fresh timestamped run directory instead of using the latest",
    )

    limits = parser.add_argument_group("limits (for testing on smaller batches)")
    limits.add_argument(
        "--max-repos", type=int, default=1000, help="Limit number of repos processed"
    )
    limits.add_argument(
        "--repo-min-stars",
        type=int,
        default=0,
        help="Only include repos with at least this many stars",
    )
    limits.add_argument(
        "--repo-min-contributors",
        type=int,
        default=0,
        help="Drop repos with fewer than this many contributors",
    )
    limits.add_argument(
        "--max-generate",
        type=int,
        default=None,
        help="Limit files sent to the LLM for generation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MINING_WORKERS,
        help="Number of threads for concurrent repo cloning/mining",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny run: sets all limits to small values for a quick end-to-end test",
    )

    args = parser.parse_args()

    no_stage_selected = (
        not args.collect and not args.generate and not args.report
    )
    if no_stage_selected:
        args.collect = True
        args.generate = True
        args.report = True

    if args.smoke_test:
        args.max_repos = 10
        args.max_generate = 3

    return args


def main():
    args = parse_args()
    run_dir = resolve_run_directory(create_new_run=args.new_run)
    logging.info("Using run directory: %s", run_dir)

    if args.collect:
        collect_dataset(
            run_dir,
            max_repos=args.max_repos,
            repo_min_stars=args.repo_min_stars,
            num_workers=args.workers,
        )
    if args.generate:
        generate_comments_for_dataset(run_dir, limit=args.max_generate)
    if args.report:
        generate_report(run_dir, args.report_dataset)


if __name__ == "__main__":
    main()
