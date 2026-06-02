from collect import collect_dataset
from analyse import analyse_dataset
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
    parser.add_argument("--analyse", action="store_true", help="Analyse collected data")
    parser.add_argument(
        "--report", action="store_true", help="Build dataset statistics and graphs"
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a fresh timestamped run directory instead of using the latest",
    )

    limits = parser.add_argument_group("limits (for testing on smaller batches)")
    limits.add_argument(
        "--max-repos", type=int, default=None, help="Limit number of repos processed"
    )
    limits.add_argument(
        "--max-commits-per-repo",
        type=int,
        default=None,
        help="Limit commits fetched per repo",
    )
    limits.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Limit total commit-detail fetches",
    )
    limits.add_argument(
        "--max-files", type=int, default=None, help="Limit total file-content fetches"
    )
    limits.add_argument(
        "--max-generate",
        type=int,
        default=None,
        help="Limit files sent to the LLM for generation",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny run: sets all limits to small values for a quick end-to-end test",
    )

    args = parser.parse_args()

    no_stage_selected = (
        not args.collect
        and not args.generate
        and not args.analyse
        and not args.report
    )
    if no_stage_selected:
        args.collect = True
        args.generate = True
        args.analyse = True
        args.report = True

    if args.smoke_test:
        args.max_repos = args.max_repos or 10
        args.max_commits_per_repo = args.max_commits_per_repo or 100
        args.max_commits = args.max_commits or None
        args.max_files = args.max_files or None
        args.max_generate = args.max_generate or 3

    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    run_dir = resolve_run_directory(create_new_run=args.new_run)
    logging.info("Using run directory: %s", run_dir)

    if args.collect:
        collect_dataset(
            run_dir,
            max_repos=args.max_repos,
            max_commits_per_repo=args.max_commits_per_repo,
            max_commits=args.max_commits,
            max_files=args.max_files,
        )
    if args.generate:
        generate_comments_for_dataset(run_dir, limit=args.max_generate)
    if args.analyse:
        analyse_dataset(run_dir)
    if args.report:
        generate_report(run_dir)


if __name__ == "__main__":
    main()
