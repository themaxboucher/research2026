import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

from llms import openrouter, transformers
from storage import append_to_jsonl, iter_from_jsonl, save_to_jsonl
from comments import is_machine_directive_comment


SOURCE_FILENAME = "repo_files_sample"

# These are the approaches we use to prompt the LLMs. Different approaches produce
# different types of outputs (e.g. generate only the comment vs regenerate the
# whole code with comments added).
APPROACHES = ("location", "regenerate")

GENERATIONS_DIRNAME = "generations"
MANIFEST_FILENAME = "generation.json"
LOCATION_FILENAME = "location_generated"
REGENERATE_FILENAME = "regenerate_generated"
NOTES_FILENAME = "review_notes.json"


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
) -> dict:
    manifest = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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


MODEL_PROFILES = {
    "local": ModelProfile(
        model_names=[
            "meta-llama/llama-3.1-8b-instruct",
            "qwen/qwen-2.5-7b-instruct",
            "deepseek/deepseek-v4-pro",
            "z-ai/glm-5.2",
            "openai/gpt-5.6-luna-pro",
        ],
        get_completion=openrouter.get_completion,
    ),
    "cluster": ModelProfile(
        model_names=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        get_completion=transformers.get_completion,
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
        and not is_machine_directive_comment(comment["comment"])
        and comment.get("intent") in TARGET_INTENTS
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


def generate_comments_for_dataset(
    run_dir: Path,
    label: str | None = None,
    limit: int | None = None,
    approaches: list[str] | None = None,
) -> Path:
    # Imported here so each approach module can share this module's model
    # profile and target-comment helpers without a circular import.
    from location_generate import location_generate_for_file
    from regenerate_generate import regenerate_generate_for_file

    approaches = list(approaches or APPROACHES)
    unknown_approaches = set(approaches) - set(APPROACHES)
    if unknown_approaches:
        raise ValueError(
            f"Unknown approaches: {', '.join(sorted(unknown_approaches))}. "
            f"Expected any of: {', '.join(APPROACHES)}"
        )

    files_data = iter_from_jsonl(run_dir, SOURCE_FILENAME)

    model_profile, model_profile_name = get_model_profile()

    label = label or default_label()
    gen_dir = generation_dir(run_dir, label)
    existing_output_filenames = [
        filename
        for filename in (LOCATION_FILENAME, REGENERATE_FILENAME)
        if (gen_dir / f"{filename}.jsonl").exists()
    ]
    if existing_output_filenames:
        logging.warning(
            "Generation %r already exists at %s; overwriting its output "
            "(review notes are preserved).",
            label,
            gen_dir,
        )

    write_manifest(
        gen_dir,
        label=label,
        model_profile=model_profile_name,
        model_names=model_profile.model_names,
        config={"max_generate": limit, "approaches": approaches},
    )

    if "location" in approaches:
        save_to_jsonl([], gen_dir, LOCATION_FILENAME)
    if "regenerate" in approaches:
        save_to_jsonl([], gen_dir, REGENERATE_FILENAME)

    files_processed = 0

    for file_data in files_data:
        if not _is_eligible_file(file_data):
            continue

        if limit is not None and files_processed >= limit:
            break

        filepath = file_data.get("new_path")
        logging.info(
            "Generating comments for %s (file %d/%s)",
            filepath,
            files_processed + 1,
            limit if limit is not None else "?",
        )

        if "regenerate" in approaches:
            scope_records = regenerate_generate_for_file(file_data, model_profile)
            append_to_jsonl(scope_records, gen_dir, REGENERATE_FILENAME)

        if "location" in approaches:
            location_record = location_generate_for_file(file_data, model_profile)
            append_to_jsonl([location_record], gen_dir, LOCATION_FILENAME)

        files_processed += 1

    return gen_dir
