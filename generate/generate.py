import argparse
import itertools
import json
import logging
import os
import re
import subprocess
import tokenize
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

from generate.llms import openrouter
from generate.llms import transformers
from collect.dataset import require_latest_dataset_directory
from storage import (
    append_to_jsonl,
    drop_trailing_records,
    iter_from_jsonl,
    merge_jsonl_shards,
    save_to_jsonl,
    truncate_broken_tail,
)
from generate.comments import get_comments_from_file, is_machine_directive_comment


SOURCE_FILENAME = "dataset_sample"

# These are the approaches we use to prompt the LLMs. Different approaches produce
# different types of outputs (e.g. generate only the comment vs regenerate the
# whole code with comments added).
APPROACHES = ("location", "regenerate")

GENERATIONS_DIRNAME = "generations"
MANIFEST_FILENAME = "generation.json"
LOCATION_FILENAME = "location_generated"
REGENERATE_FILENAME = "regenerate_generated"
NOTES_FILENAME = "review_notes.json"

PROGRESS_FILENAME = "generation_progress"


def _file_key(record: dict) -> str:
    # `\x00` is a safe separator because it's not valid in a GitHub repo name, path, or commit hash.
    return "\x00".join(
        str(record.get(field)) for field in ("repo_name", "new_path", "commit_hash")
    )


def _completed_file_keys(gen_dir: Path, progress_filename: str) -> set[str]:
    """File keys already committed to the progress log for this generation."""
    if not (gen_dir / f"{progress_filename}.jsonl").exists():
        return set()
    return {_file_key(record) for record in iter_from_jsonl(gen_dir, progress_filename)}


def _shard_suffix(task_id: int, num_tasks: int) -> str:
    digit_width = max(len(str(num_tasks - 1)), 1)
    return f"{task_id:0{digit_width}d}"


def _sharded(filename: str, suffix: str | None) -> str:
    return f"{filename}.{suffix}" if suffix is not None else filename


def _has_shards(gen_dir: Path) -> bool:
    for filename in (LOCATION_FILENAME, REGENERATE_FILENAME, PROGRESS_FILENAME):
        if any(gen_dir.glob(f"{filename}.*.jsonl")):
            return True
    return False


def _slugify(label: str) -> str:
    """Filesystem-safe directory name for a user-supplied generation label."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-._")
    return slug or "generation"


def default_label() -> str:
    """Timestamp label used when the caller doesn't name the generation."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def generations_root(run_dir: Path) -> Path:
    return run_dir / GENERATIONS_DIRNAME


def generation_dir(run_dir: Path, label: str) -> Path:
    return generations_root(run_dir) / _slugify(label)


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if git isn't available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip() or None


def write_manifest(
    gen_dir: Path,
    label: str,
    model_profile: str | None,
    model_names: list[str],
    config: dict,
    created_at: str | None = None,
) -> dict:
    manifest = {
        "label": label,
        # Preserve the original creation time when resuming an existing generation.
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_profile": model_profile,
        "model_names": list(model_names),
        "git_commit": _git_commit(),
        "config": config,
    }
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def read_manifest(gen_dir: Path) -> dict:
    manifest_path = gen_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def list_generations(run_dir: Path) -> list[dict]:
    root = generations_root(run_dir)
    if not root.exists():
        return []

    generations: list[dict] = []
    for gen_dir in root.iterdir():
        if not gen_dir.is_dir():
            continue
        has_output = any(
            (gen_dir / f"{filename}.jsonl").exists()
            for filename in (LOCATION_FILENAME, REGENERATE_FILENAME)
        )
        if not has_output:
            continue
        manifest = read_manifest(gen_dir)
        generations.append(
            {
                "id": gen_dir.name,
                "label": manifest.get("label") or gen_dir.name,
                "created_at": manifest.get("created_at"),
                "model_profile": manifest.get("model_profile"),
                "model_names": manifest.get("model_names") or [],
                "config": manifest.get("config") or {},
                "dir": gen_dir,
            }
        )

    # Newest first. Missing timestamps sort last so named-but-manifestless
    # generations don't crowd out dated ones.
    generations.sort(key=lambda gen: gen.get("created_at") or "", reverse=True)
    return generations


class ModelProfile(NamedTuple):
    model_names: list[str]
    get_completion: Callable[[str, str], str]
    # Files generated in parallel. Each file also fans out one request per
    # model, so total in-flight requests ≈ concurrent_files × len(model_names).
    concurrent_files: int


MODEL_PROFILES = {
    "local": ModelProfile(
        model_names=[
            "meta-llama/llama-3.1-8b-instruct",
            "qwen/qwen-2.5-7b-instruct",
        ],
        get_completion=openrouter.get_completion,
        concurrent_files=24,
    ),
    "cluster": ModelProfile(
        model_names=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        get_completion=transformers.get_completion,
        # The GPU serializes forward passes; concurrent files would only
        # interleave and slow each other down.
        concurrent_files=1,
    ),
}
DEFAULT_MODEL_PROFILE = "local"


def get_model_profile() -> tuple[ModelProfile, str]:
    profile_name = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if profile_name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MODEL_PROFILE {profile_name!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[profile_name], profile_name


def _has_at_least_one_alpha_char(s: str) -> bool:
    """Check if a string contains at least one alphabetic character."""
    return any(char.isalpha() for char in s)


def _is_visual_separator_comment(comment: str) -> bool:
    """Check if a comment is a visual separator (e.g., a line of dashes or equals)."""
    # A separator is a run of at least 5 identical punctuation characters,
    # such as "-----", "=====", or "*****".
    run_length = 0
    previous_char = ""
    for char in comment:
        if char.isalnum() or char.isspace() or char == "#":
            run_length = 0
            previous_char = ""
            continue
        if char == previous_char:
            run_length += 1
        else:
            run_length = 1
            previous_char = char
        if run_length >= 5:
            return True
    return False


def _target_comments(source_file: dict) -> list[dict]:
    TARGET_COMMENT_TYPES = {"inline", "block"}
    TARGET_COMMENT_STATUSES = {"added"}
    # `None` means the comment's intent was never labeled; it gets the generic instruction
    TARGET_INTENTS = {"what", "why", "how", None}
    return [
        comment
        for comment in (source_file.get("comments") or [])
        if comment.get("type") in TARGET_COMMENT_TYPES
        and comment.get("status") in TARGET_COMMENT_STATUSES
        and comment.get("comment") is not None
        and comment.get("intent") in TARGET_INTENTS
        and not is_machine_directive_comment(comment["comment"])
        and _has_at_least_one_alpha_char(comment.get("comment", ""))
        and not _is_visual_separator_comment(comment.get("comment", ""))
        and comment.get("comment", "").isascii()  # Exclude non-english comments
    ]


def _is_ai_authored_file(source_file: dict) -> bool:
    AI_AUTHORED_IDENTIFIERS = {
        # Anthropic — Claude Code
        "Co-authored-by: Claude",
        "noreply@anthropic.com",
        "Generated with Claude Code",
        # GitHub Copilot — coding agent / Copilot CLI
        "Co-authored-by: Copilot",
        "Copilot@users.noreply.github.com",  # e.g. 198982749+Copilot@users.noreply.github.com
        "Copilot[bot]@users.noreply.github.com",
        # Cursor — background/cloud agent
        "Co-authored-by: Cursor",
        "cursoragent@cursor.com",
        "cursoragent@users.noreply.github.com",
        "Made-with: Cursor",
        # OpenAI — Codex CLI / Codex cloud
        "Co-authored-by: Codex",
        "noreply@openai.com",
        "chatgpt-codex-connector[bot]",
        # Google — Gemini CLI / Gemini Code Assist
        "Co-authored-by: Gemini",
        "gemini-code-assist[bot]",
        "gemini-cli@users.noreply.github.com",
        "gemini-cli-agent@google.com",
        # Aider
        "Co-authored-by: aider",
        "noreply@aider.chat",
        "aider@aider.chat",
        # Cognition — Devin
        "Co-authored-by: Devin",
        "devin-ai-integration",  # also covers the devin-ai-integration[bot] account
        # Generic / cross-tool markers
        "🤖 Generated with",
        "Assisted-by:",
        "Co-authored-by: AI",
    }
    if source_file.get("commit_message") is None:
        raise ValueError(
            "Commit message is required to determine if a file is AI-authored"
        )
    commit_message = source_file.get("commit_message").lower()
    for identifier in AI_AUTHORED_IDENTIFIERS:
        if identifier.lower() in commit_message:
            return True
    return False


def _is_eligible_file(source_file: dict) -> bool:
    """Check if a file record should be included in the generation."""
    is_python_file = source_file.get("new_path", "").endswith(".py")
    if not is_python_file:
        return False

    ELIGIBLE_CHANGE_TYPES = {"MODIFY"}
    is_valid_change_type = source_file.get("change_type") in ELIGIBLE_CHANGE_TYPES
    if not is_valid_change_type:
        return False

    has_previous_source_code = source_file.get("previous_source_code") is not None
    if not has_previous_source_code:
        return False

    has_target_comments = bool(_target_comments(source_file))
    if not has_target_comments:
        return False

    has_commit_message = source_file.get("commit_message") is not None
    if not has_commit_message:
        return False

    is_ai_authored = _is_ai_authored_file(source_file)
    if is_ai_authored:
        return False

    return True


def _with_parsed_comments(files_data):
    """Annotate each streamed file record with its parsed code comments."""
    for file_data in files_data:
        file_data["comments"] = []
        source_code = file_data.get("source_code")
        previous_source_code = file_data.get("previous_source_code")
        # Records missing either side (e.g. added or deleted files) are
        # filtered out by _is_eligible_file before their comments are read.
        if source_code is None or previous_source_code is None:
            yield file_data
            continue
        try:
            file_data["comments"] = get_comments_from_file(
                source_code, previous_source_code
            )
        except (tokenize.TokenError, SyntaxError) as e:
            logging.warning(
                "Failed to parse comments for %s: %s", file_data.get("new_path"), e
            )
            file_data["error"] = str(e)
        yield file_data


def _generate_outputs_for_file(
    file_data: dict,
    model_profile: ModelProfile,
    approaches: list[str],
) -> list[tuple[str, list[dict]]]:
    # Imported here so each approach module can share this module's model
    # profile and target-comment helpers without a circular import.
    from generate.location_generate import location_generate_for_file
    from generate.regenerate_generate import regenerate_generate_for_file

    outputs: list[tuple[str, list[dict]]] = []
    if "regenerate" in approaches:
        scope_records = regenerate_generate_for_file(file_data, model_profile)
        outputs.append((REGENERATE_FILENAME, scope_records))
    if "location" in approaches:
        location_record = location_generate_for_file(file_data, model_profile)
        outputs.append((LOCATION_FILENAME, [location_record]))
    return outputs


def _validate_approaches(approaches: list[str] | None) -> list[str]:
    approaches = list(approaches or APPROACHES)
    unknown_approaches = set(approaches) - set(APPROACHES)
    if unknown_approaches:
        raise ValueError(
            f"Unknown approaches: {', '.join(sorted(unknown_approaches))}. "
            f"Expected any of: {', '.join(APPROACHES)}"
        )
    return approaches


def _output_filenames(approaches: list[str], suffix: str | None) -> list[str]:
    return [
        _sharded(filename, suffix)
        for filename, approach in (
            (LOCATION_FILENAME, "location"),
            (REGENERATE_FILENAME, "regenerate"),
        )
        if approach in approaches
    ]


def _run_generation(
    run_dir: Path,
    gen_dir: Path,
    label: str,
    model_profile: ModelProfile,
    approaches: list[str],
    limit: int | None,
    task_id: int | None = None,
    num_tasks: int | None = None,
) -> Path:
    in_jobs_array = task_id is not None and num_tasks is not None
    suffix = _shard_suffix(task_id, num_tasks) if in_jobs_array else None
    progress_filename = _sharded(PROGRESS_FILENAME, suffix)
    output_filenames = _output_filenames(approaches, suffix)

    concurrent_files = model_profile.concurrent_files

    # Resume an interrupted run for this label rather than overwriting it
    truncate_broken_tail(gen_dir, progress_filename)
    completed_keys = _completed_file_keys(gen_dir, progress_filename)
    for filename in output_filenames:
        truncate_broken_tail(gen_dir, filename)
        drop_trailing_records(
            gen_dir,
            filename,
            lambda record: _file_key(record) not in completed_keys,
        )
    if completed_keys:
        logging.info(
            "Resuming generation %r at %s (%d files already done).",
            label,
            gen_dir,
            len(completed_keys),
        )

    # Create empty output files for a fresh run
    for filename in output_filenames:
        if not (gen_dir / f"{filename}.jsonl").exists():
            save_to_jsonl([], gen_dir, filename)

    files_data = _with_parsed_comments(iter_from_jsonl(run_dir, SOURCE_FILENAME))

    eligible_files = (
        file_data for file_data in files_data if _is_eligible_file(file_data)
    )
    if limit is not None:
        eligible_files = itertools.islice(eligible_files, limit)
    # Every array task streams the same source and applies the same filter and
    # limit, so striding over the eligible files deterministically partitions
    # them across tasks (mirrors collection's repos[task_id::num_tasks]).
    if in_jobs_array:
        eligible_files = itertools.islice(eligible_files, task_id, None, num_tasks)
    # Filter out files already completed by an earlier run of this generation. This
    # runs after `limit` so the target set matches a fresh run's first `limit`
    # eligible files. We just don't redo the ones already finished.
    eligible_files = (
        file_data
        for file_data in eligible_files
        if _file_key(file_data) not in completed_keys
    )

    # Count files already done so progress logging reflects the whole run.
    files_processed = len(completed_keys)

    # Files are generated concurrently but submitted through a window of at
    # most `concurrent_files`, so the source JSONL streams instead of being
    # held in memory. All appends happen here on the main thread.
    with ThreadPoolExecutor(max_workers=concurrent_files) as executor:
        in_flight: dict[Future, dict] = {}

        def submit_next_file() -> bool:
            file_data = next(eligible_files, None)
            if file_data is None:
                return False
            future = executor.submit(
                _generate_outputs_for_file, file_data, model_profile, approaches
            )
            in_flight[future] = {
                "repo_name": file_data.get("repo_name"),
                "new_path": file_data.get("new_path"),
                "commit_hash": file_data.get("commit_hash"),
            }
            return True

        for _ in range(concurrent_files):
            if not submit_next_file():
                break

        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                file_key = in_flight.pop(future)
                try:
                    outputs = future.result()
                except Exception as error:
                    # A failed file is recorded as done (with its error) so one
                    # pathological input can't wedge the task or spin forever
                    # on resume; it just produces no output records.
                    logging.warning(
                        "Generation failed for %s: %s", file_key["new_path"], error
                    )
                    append_to_jsonl(
                        [{**file_key, "error": str(error)}],
                        gen_dir,
                        progress_filename,
                    )
                else:
                    for output_filename, records in outputs:
                        append_to_jsonl(
                            records, gen_dir, _sharded(output_filename, suffix)
                        )
                    # Log only after every approach's output is on disk, so a
                    # crash mid-write leaves this file uncommitted and it is redone.
                    append_to_jsonl(
                        [{**file_key, "error": None}], gen_dir, progress_filename
                    )
                    logging.info(
                        "Generated comments for %s (file %d/%s)",
                        file_key["new_path"],
                        files_processed + 1,
                        limit if limit is not None and not in_jobs_array else "?",
                    )
                files_processed += 1
                submit_next_file()

    logging.info(
        "Finished generation %r at %s (%d files generated).",
        label,
        gen_dir,
        files_processed,
    )
    return gen_dir


def generate_comments_for_dataset(
    run_dir: Path,
    label: str | None = None,
    limit: int | None = None,
    approaches: list[str] | None = None,
) -> Path:
    """Generate on a single machine, writing unsharded output files."""
    approaches = _validate_approaches(approaches)

    model_profile, model_profile_name = get_model_profile()

    label = label or default_label()
    gen_dir = generation_dir(run_dir, label)

    existing_manifest = read_manifest(gen_dir)
    if (existing_manifest.get("config") or {}).get("num_tasks") is not None or (
        _has_shards(gen_dir)
    ):
        raise RuntimeError(
            f"Generation {label!r} was produced by a job array; resume it with "
            "generate/scripts/generate-submit.sh or choose a new label"
        )

    write_manifest(
        gen_dir,
        label=label,
        model_profile=model_profile_name,
        model_names=model_profile.model_names,
        config={
            "max_generate": limit,
            "approaches": approaches,
            "concurrent_files": model_profile.concurrent_files,
        },
        created_at=existing_manifest.get("created_at"),
    )

    return _run_generation(run_dir, gen_dir, label, model_profile, approaches, limit)


def prepare_generation(
    run_dir: Path,
    label: str | None,
    num_tasks: int | None,
    limit: int | None,
    approaches: list[str] | None,
) -> tuple[str, int]:
    approaches = _validate_approaches(approaches)

    source_path = run_dir / f"{SOURCE_FILENAME}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(
            f"No sample file at {source_path}. Run collect/sample.py on this run first."
        )

    model_profile, model_profile_name = get_model_profile()

    label = label or default_label()
    gen_dir = generation_dir(run_dir, label)
    existing_manifest = read_manifest(gen_dir)

    if existing_manifest:
        existing_config = existing_manifest.get("config") or {}
        existing_num_tasks = existing_config.get("num_tasks")
        if existing_num_tasks is None:
            raise RuntimeError(
                f"Generation {label!r} was generated without a job array; "
                "choose a new label"
            )
        if num_tasks is not None and num_tasks != existing_num_tasks:
            raise ValueError(
                f"Generation {label!r} was created with --num-tasks "
                f"{existing_num_tasks}; resuming with {num_tasks} would "
                "reshuffle which task owns which file"
            )
        num_tasks = existing_num_tasks
        approaches = existing_config.get("approaches") or approaches
        limit = existing_config.get("max_generate")

    if num_tasks is None:
        raise ValueError("--num-tasks is required for a new array generation")
    if num_tasks < 1:
        raise ValueError(f"--num-tasks must be >= 1, got {num_tasks}")

    write_manifest(
        gen_dir,
        label=label,
        model_profile=model_profile_name,
        model_names=model_profile.model_names,
        config={
            "max_generate": limit,
            "approaches": approaches,
            "num_tasks": num_tasks,
        },
        created_at=existing_manifest.get("created_at"),
    )
    return label, num_tasks


def generate_comments_for_task(run_dir: Path, label: str, task_id: int) -> Path:
    """Generate one array task's stride of the eligible files, into shard files."""
    gen_dir = generation_dir(run_dir, label)
    manifest = read_manifest(gen_dir)
    if not manifest:
        raise RuntimeError(
            f"No manifest at {gen_dir / MANIFEST_FILENAME}; submit the array "
            "through generate/scripts/generate-submit.sh so the manifest is written first"
        )

    config = manifest.get("config") or {}
    num_tasks = config.get("num_tasks")
    if num_tasks is None:
        raise RuntimeError(
            f"Generation {label!r} was not prepared for a job array; choose a new label"
        )
    if not 0 <= task_id < num_tasks:
        raise ValueError(
            f"--task-id {task_id} is out of range for --num-tasks {num_tasks}"
        )

    model_profile, model_profile_name = get_model_profile()
    manifest_profile = manifest.get("model_profile")
    if manifest_profile is not None and manifest_profile != model_profile_name:
        raise RuntimeError(
            f"Generation {label!r} was prepared with MODEL_PROFILE="
            f"{manifest_profile!r} but this task is running with "
            f"{model_profile_name!r}"
        )

    return _run_generation(
        run_dir,
        gen_dir,
        label,
        model_profile,
        approaches=config.get("approaches") or list(APPROACHES),
        limit=config.get("max_generate"),
        task_id=task_id,
        num_tasks=num_tasks,
    )


def finalize_generation(run_dir: Path, label: str) -> Path:
    """Merge the per-task shard files into the unsharded files readers expect."""
    gen_dir = generation_dir(run_dir, label)

    for filename in (LOCATION_FILENAME, REGENERATE_FILENAME, PROGRESS_FILENAME):
        # Skip filenames with no shards so an approach that never ran doesn't
        # get an empty merged file (which readers would take as real output).
        if not any(gen_dir.glob(f"{filename}.*.jsonl")):
            continue
        shard_count = merge_jsonl_shards(gen_dir, filename)
        logging.info("Merged %d %s shards", shard_count, filename)

    completed_files = 0
    failed_files = 0
    if (gen_dir / f"{PROGRESS_FILENAME}.jsonl").exists():
        for record in iter_from_jsonl(gen_dir, PROGRESS_FILENAME):
            if record.get("error"):
                failed_files += 1
            else:
                completed_files += 1
    logging.info(
        "Finalized generation %r: %d files generated, %d failed",
        label,
        completed_files,
        failed_files,
    )
    return gen_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM comments for a collected, sampled run. With no "
        "stage flag, generates on a single machine; --prepare/--finalize and "
        "--task-id drive the SLURM job array (see generate/scripts/generate-submit.sh)."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory to generate for (defaults to the latest run)",
    )
    parser.add_argument(
        "--generation",
        type=str,
        default=None,
        help="Label for this generation, written under "
        "runs/<run>/generations/<label>/ (defaults to a timestamp). Re-running a "
        "label resumes it, skipping files already generated.",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default=",".join(APPROACHES),
        help="Comma-separated generation approaches to run: 'location' prompts "
        "for a comment at a given spot, 'regenerate' has the model rewrite each "
        "scope with comments added (default: both)",
    )
    parser.add_argument(
        "--max-generate",
        type=int,
        default=None,
        help="Limit files sent to the LLM for generation",
    )

    stages = parser.add_argument_group("job array (HPC partitioning)")
    stages.add_argument(
        "--prepare",
        action="store_true",
        help="Write the generation manifest for a job array and print RUN_DIR, "
        "GENERATION and NUM_TASKS for generate/scripts/generate-submit.sh",
    )
    stages.add_argument(
        "--finalize",
        action="store_true",
        help="Merge a generation's per-task output shards into single files",
    )
    stages.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="This task's index in the job array. Generates only this task's "
        "share of the eligible files into its own sharded files",
    )
    stages.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total number of tasks in the job array; --prepare records it in "
        "the manifest so every task strides the same way",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else require_latest_dataset_directory()
    logging.info("Using run directory: %s", run_dir)

    approaches = [
        approach.strip() for approach in args.approaches.split(",") if approach.strip()
    ]

    if args.prepare:
        label, num_tasks = prepare_generation(
            run_dir,
            label=args.generation,
            num_tasks=args.num_tasks,
            limit=args.max_generate,
            approaches=approaches,
        )
        # generate/scripts/generate-submit.sh uses these prints to parse the array parameters
        print(f"RUN_DIR={run_dir}")
        print(f"GENERATION={label}")
        print(f"NUM_TASKS={num_tasks}")
        return

    if args.finalize:
        if not args.generation:
            raise SystemExit("--finalize requires --generation")
        finalize_generation(run_dir, args.generation)
        return

    if args.task_id is not None:
        if not args.generation:
            raise SystemExit("--generation is required when generating with --task-id")
        generate_comments_for_task(run_dir, label=args.generation, task_id=args.task_id)
        return

    generate_comments_for_dataset(
        run_dir,
        label=args.generation,
        limit=args.max_generate,
        approaches=approaches,
    )


if __name__ == "__main__":
    main()
