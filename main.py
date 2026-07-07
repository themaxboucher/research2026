from collect import (
    collect_dataset,
    finalize_collection,
    prepare_collection,
)
from generate import generate_comments_for_dataset
from report import generate_report
from runs import resolve_run_directory
import argparse
import logging
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Search GitHub once, cache the repo list, and print RUN_DIR and "
        "NUM_TASKS for the job array",
    )
    parser.add_argument(
        "--collect", action="store_true", help="Collect data from GitHub"
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Merge the per-task output shards into single files",
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate comments for collected data"
    )
    parser.add_argument(
        "--generation",
        type=str,
        default=None,
        help="Label for this generation, written under "
        "runs/<run>/generations/<label>/ (defaults to a timestamp). Re-running a "
        "label overwrites its output but keeps its review notes",
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
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Use this exact run directory instead of picking one "
        "automatically, so every array task shares one run",
    )

    array = parser.add_argument_group("job array (HPC partitioning)")
    array.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the job array. With --num-tasks, mines only "
        "this task's share of repos into its own sharded files",
    )
    array.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the job array. With --task-id, splits "
        "the repos evenly across tasks",
    )
    array.add_argument(
        "--repos-per-task",
        type=int,
        default=10,
        help="Repos per array task; --prepare uses this to decide how many "
        "tasks to create",
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
        "--smoke-test",
        action="store_true",
        help="Tiny run: sets all limits to small values for a quick end-to-end test",
    )

    args = parser.parse_args()

    no_stage_selected = not any(
        (args.prepare, args.collect, args.finalize, args.generate, args.report)
    )
    if no_stage_selected:
        args.collect = True
        args.generate = True
        args.report = True

    if args.smoke_test:
        args.max_repos = 10
        args.max_generate = 10

    return args


def main():
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = resolve_run_directory(create_new_run=(args.new_run or args.prepare))
    logging.info("Using run directory: %s", run_dir)

    if args.prepare:
        num_tasks = prepare_collection(
            run_dir,
            max_repos=args.max_repos,
            repo_min_stars=args.repo_min_stars,
            repos_per_task=args.repos_per_task,
        )
        # submit.sh uses these prints to parse the RUN_DIR and NUM_TASKS
        print(f"RUN_DIR={run_dir}")
        print(f"NUM_TASKS={num_tasks}")
        return

    if args.collect:
        collect_dataset(
            run_dir,
            task_id=args.task_id,
            num_tasks=args.num_tasks,
            max_repos=args.max_repos,
            repo_min_stars=args.repo_min_stars,
        )
    if args.finalize:
        finalize_collection(run_dir)
    if args.generate:
        generate_comments_for_dataset(
            run_dir, label=args.generation, limit=args.max_generate
        )
    if args.report:
        generate_report(run_dir, args.report_dataset)


if __name__ == "__main__":
    main()
