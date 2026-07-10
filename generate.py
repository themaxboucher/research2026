import ast
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Callable, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llms import openrouter, transformers
from storage import append_to_jsonl, iter_from_jsonl, save_to_jsonl
from comments import extract_comments, is_machine_directive_comment
from prompt import build_prompt
from scopes import (
    enclosing_scope_name as _enclosing_scope_name,
    local_scope_bounds as _local_scope_bounds,
    scope_code as _scope_code,
)
import generations


SOURCE_FILENAME = "repo_files_sample"
GENERATED_DATASET_FILENAME = "files_generated"


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


def get_model_profile() -> ModelProfile:
    profile = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MODEL_PROFILE {profile!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[profile]


def _diff_region_bounds(
    source_code: str, source_lines: list[str], anchor_line: int
) -> tuple[int, int]:
    """Line bounds (1-indexed, inclusive) of the local scope enclosing
    `anchor_line`. When the change isn't inside a function or class, fall back
    to a window of up to MAX_LINE_COUNT lines centered on the anchor."""
    MAX_LINE_COUNT = 500

    qualified_name = _enclosing_scope_name(source_code, anchor_line)
    if qualified_name is not None:
        scope_bounds = _local_scope_bounds(source_lines, source_code, qualified_name)
        if scope_bounds is not None:
            return scope_bounds

    top = bottom = anchor_line
    while (bottom - top + 1) < min(MAX_LINE_COUNT, len(source_lines)):
        if top > 1:
            top -= 1
        if bottom < len(source_lines):
            bottom += 1
    return top, bottom


_HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _iter_hunks(diff: str):
    current: dict | None = None
    for line in diff.splitlines(keepends=True):
        match = _HUNK_HEADER_PATTERN.match(line)
        if match:
            if current is not None:
                yield current
            # Anything after the closing `@@` is the function-context hint
            # git adds (e.g. " def foo():"). Keep it so the LLM gets the same
            # signal git would have shown.
            header_suffix = line[match.end() :].rstrip("\n")
            current = {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2)) if match.group(2) else 1,
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4)) if match.group(4) else 1,
                "header_suffix": header_suffix,
                "body": [],
            }
        elif current is not None:
            current["body"].append(line)
    if current is not None:
        yield current


def _target_line_code(body_line: str, comment_data: dict) -> str | None:
    """The comment-free code that should remain on a target `+` line. A block
    comment line is wholly comment, so nothing remains (None). An inline comment
    sits on a code line that may itself have changed, so keep the code."""
    if comment_data["type"] != "inline":
        return None
    anchor = comment_data.get("anchor")
    if anchor is not None:
        return anchor
    return body_line[1:].split("#", 1)[0].rstrip()


def _stripped_hunk_body(hunk: dict, comment_data: dict) -> list[str]:
    """Return the hunk body with the human's target comment removed. A block
    target `+` line is dropped and its paired `-` reverts to context. An inline
    target keeps its code change (comment stripped), collapsing to context only
    when the code itself is unchanged. Surrounding changes keep their semantics —
    each `+` consumes its positionally paired `-` (FIFO), so a non-target `+`
    can't steal a dash that belongs to a later target `+`."""
    target_start_line = comment_data["start_line"]
    target_end_line = comment_data["end_line"]
    output_body: list[str] = []
    pending_dashes: list[str] = []  # `-` lines awaiting pairing with `+` lines
    new_line = hunk["new_start"]

    for body_line in hunk["body"]:
        if not body_line or body_line[0] == "\\":
            # Markers like "\ No newline at end of file" stay as-is.
            output_body.append(body_line)
            continue

        prefix = body_line[0]
        if prefix == " ":
            output_body.extend(pending_dashes)
            pending_dashes = []
            output_body.append(body_line)
            new_line += 1
        elif prefix == "-":
            pending_dashes.append(body_line)
        elif prefix == "+":
            line_is_target = target_start_line <= new_line <= target_end_line
            paired_dash = pending_dashes.pop(0) if pending_dashes else None
            if line_is_target:
                code = _target_line_code(body_line, comment_data)
                # Compare the code either side of the change, ignoring comments.
                # A naive split can only over-report a change (a `#` in a string),
                # which is safe: it shows a `-`/`+` pair rather than hiding a change.
                old_code = (
                    None
                    if paired_dash is None
                    else paired_dash[1:].split("#", 1)[0].rstrip()
                )
                code_is_unchanged = (
                    old_code is not None and old_code == (code or "").rstrip()
                )
                if code is None or code_is_unchanged:
                    # Nothing left to show, or the comment was the only change:
                    # fold back to a single context line where one existed.
                    if paired_dash is not None:
                        output_body.append(" " + paired_dash[1:])
                else:
                    if paired_dash is not None:
                        output_body.append(paired_dash)
                    output_body.append("+" + code + "\n")
            else:
                if paired_dash is not None:
                    output_body.append(paired_dash)
                output_body.append(body_line)
            new_line += 1
        else:
            output_body.append(body_line)

    output_body.extend(pending_dashes)
    return output_body


def _scope_diff(diff: str, source_code: str, comment_data: dict) -> str | None:
    """Build one unified-diff hunk whose context spans the entire local scope
    enclosing the changed comment. Every change hunk within that scope is merged
    in, the gaps between them padded with the scope's unchanged lines, and the
    human's target comment stripped out. Returns None when no hunk overlaps the
    scope (the target comment can't be located in the change)."""
    source_lines = source_code.splitlines()

    region_start, region_end = _diff_region_bounds(
        source_code, source_lines, comment_data["start_line"]
    )

    overlapping = []
    for hunk in _iter_hunks(diff):
        new_end = hunk["new_start"] + max(hunk["new_count"] - 1, 0)
        overlaps_region = hunk["new_start"] <= region_end and region_start <= new_end
        if overlaps_region:
            overlapping.append(hunk)
    if not overlapping:
        return None
    overlapping.sort(key=lambda hunk: hunk["new_start"])

    # Widen the region so it fully contains every hunk we merge in — a hunk's
    # few lines of git context may spill just past the scope boundary.
    region_start = min(region_start, overlapping[0]["new_start"])
    last_hunk = overlapping[-1]
    last_hunk_end = last_hunk["new_start"] + max(last_hunk["new_count"] - 1, 0)
    region_end = min(max(region_end, last_hunk_end), len(source_lines))

    body: list[str] = []
    new_cursor = region_start  # Next new-side line not yet emitted.
    for hunk in overlapping:
        for line_no in range(new_cursor, hunk["new_start"]):
            body.append(" " + source_lines[line_no - 1] + "\n")
        body.extend(_stripped_hunk_body(hunk, comment_data))
        # The hunk occupies its full committed new-side span even where the
        # target comment was dropped, so advance by the original count to keep
        # the padding from re-emitting those lines.
        new_cursor = hunk["new_start"] + hunk["new_count"]
    for line_no in range(new_cursor, region_end + 1):
        body.append(" " + source_lines[line_no - 1] + "\n")

    old_start = overlapping[0]["old_start"] - (
        overlapping[0]["new_start"] - region_start
    )
    old_count = sum(1 for line in body if line and line[0] in " -")
    new_count = sum(1 for line in body if line and line[0] in " +")
    header = (
        f"@@ -{old_start},{old_count} +{region_start},{new_count} @@"
        f"{overlapping[0].get('header_suffix', '')}\n"
    )
    return header + "".join(body)


def _reverted_comment_text(target: dict, old_comments: list[dict]) -> str | None:
    for old_comment in old_comments:
        if (
            old_comment["type"] == target["type"]
            and old_comment.get("anchor") == target.get("anchor")
            and old_comment.get("comment") != target.get("comment")
        ):
            return old_comment.get("comment")
    return None


_MAX_COMMENT_LINE_LENGTH = 80


def _wrap_comment_line(line: str) -> list[str]:
    if len(line) <= _MAX_COMMENT_LINE_LENGTH:
        return [line]
    hash_count = len(line) - len(line.lstrip("#"))
    prefix = line[:hash_count] + " "
    content = line[hash_count:].strip()
    wrapped = textwrap.wrap(
        content,
        width=_MAX_COMMENT_LINE_LENGTH,
        initial_indent=prefix,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [line]


def _normalize_block_comment(text: str) -> list[str]:
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line.startswith("#")]
    lines = lines or ["# " + text.strip().lstrip("#").strip()]
    return [wrapped for line in lines for wrapped in _wrap_comment_line(line)]


def _normalize_inline_comment(text: str) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not line.startswith("#"):
        line = "# " + line.lstrip("#").strip()
    return line


def code_line_indentation(line: str) -> str:
    index_of_first_non_whitespace = len(line) - len(line.lstrip())
    return line[:index_of_first_non_whitespace]


def _replace_inline_comment(line: str, anchor: str | None, comment_text: str) -> str:
    naive_anchor_fallback = line.split("#", 1)[0].rstrip()
    anchor = anchor or naive_anchor_fallback
    two_spaces = "  "
    return anchor + two_spaces + _normalize_inline_comment(comment_text)


def _apply_new_comment(
    source_code: str, human_comment_data: dict, generated_comment_text: str
) -> str:
    source_code_lines = source_code.split("\n")
    start_index = human_comment_data["start_line"] - 1

    is_inline_comment = human_comment_data["type"] == "inline"
    if is_inline_comment:
        source_code_lines[start_index] = _replace_inline_comment(
            source_code_lines[start_index],
            human_comment_data.get("anchor"),
            generated_comment_text,
        )
        return "\n".join(source_code_lines)

    indentation = code_line_indentation(source_code_lines[start_index])
    block_lines = [
        indentation + comment_line
        for comment_line in _normalize_block_comment(generated_comment_text)
    ]
    source_code_lines[start_index : human_comment_data["end_line"]] = block_lines
    return "\n".join(source_code_lines)


_CODE_FENCE_PATTERN = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)
_XML_WRAPPER_PATTERN = re.compile(
    r"\A<(?P<tag>[A-Za-z][\w-]*)(?:\s[^>]*)?>(?P<body>.*)</(?P=tag)>\Z", re.DOTALL
)


def _strip_output_wrappers(text: str) -> str:
    """Unwrap a model response that arrives fenced in a markdown code block or
    an XML tag, leaving just the comment text. Applied repeatedly so nested
    wrappers (e.g. a code fence inside a tag) all peel away."""
    previous = None
    while text != previous:
        previous = text
        text = text.strip()
        fence_match = _CODE_FENCE_PATTERN.match(text)
        if fence_match:
            text = fence_match.group("body")
            continue
        xml_match = _XML_WRAPPER_PATTERN.match(text)
        if xml_match:
            text = xml_match.group("body")
    return text.strip()


def _generate_comment_with_model(
    prompt: str,
    filepath: str,
    source_code: str,
    comment_data: dict,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict:
    raw_response = get_completion(model_name, prompt)
    try:
        comment_text = _strip_output_wrappers(raw_response)
        if not comment_text:
            raise ValueError("Model returned an empty comment")
        new_source_code = _apply_new_comment(source_code, comment_data, comment_text)
        ast.parse(new_source_code)
    except Exception as error:
        logging.warning(
            "Failed to generate comment in %s with model %s: %s",
            filepath,
            model_name,
            error,
        )
        return {
            "model": model_name,
            "raw_response": raw_response,
            "comment_text": None,
            "new_source_code": None,
            "error": str(error),
        }
    return {
        "model": model_name,
        "raw_response": raw_response,
        "comment_text": comment_text,
        "new_source_code": new_source_code,
        "error": None,
    }


def _run_models_concurrently(
    prompt: str,
    filepath: str,
    source_code: str,
    comment_data: dict,
    model_profile: ModelProfile,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(model_profile.model_names)) as executor:
        futures = [
            executor.submit(
                _generate_comment_with_model,
                prompt,
                filepath,
                source_code,
                comment_data,
                model_name,
                model_profile.get_completion,
            )
            for model_name in model_profile.model_names
        ]
        return [future.result() for future in as_completed(futures)]


def _comment_generation(
    file_data: dict,
    comment_data: dict,
    model_profile: ModelProfile,
) -> dict:

    source_code = file_data["source_code"]
    diff = file_data["diff"]
    previous_source_code = file_data["previous_source_code"]
    filepath = file_data["new_path"]
    intent = comment_data["intent"]

    if comment_data["status"] == "modified":
        previous_comments = extract_comments(
            previous_source_code, include_docstrings=False
        )
        unmodified_comment = _reverted_comment_text(comment_data, previous_comments)
        diff_hunk = _scope_diff(diff, source_code, comment_data)
        prompt = build_prompt(
            file_data["repo_name"],
            filepath,
            comment_data,
            status="modified",
            intent=intent,
            diff_hunk=diff_hunk,
            unmodified_comment=unmodified_comment,
        )

    if comment_data["status"] == "added":
        scope_code = _scope_code(source_code, comment_data)
        prompt = build_prompt(
            file_data["repo_name"],
            filepath,
            comment_data,
            status="added",
            intent=intent,
            scope_code=scope_code,
        )

    results = _run_models_concurrently(
        prompt, filepath, source_code, comment_data, model_profile
    )
    return {
        "type": comment_data["type"],
        "status": comment_data["status"],
        "start_line": comment_data["start_line"],
        "end_line": comment_data["end_line"],
        "anchor": comment_data.get("anchor"),
        "comment": comment_data.get("comment"),
        "prompt": prompt,
        "results": results,
    }


def _target_comments(source_file: dict) -> list[dict]:
    TARGET_COMMENT_TYPES = {"inline", "block"}
    TARGET_COMMENT_STATUSES = {"added"}
    TARGET_INTENTS = {"what", "why", "how"}
    return [
        comment
        for comment in (source_file.get("comments") or [])
        if comment.get("type") in TARGET_COMMENT_TYPES
        and comment.get("status") in TARGET_COMMENT_STATUSES
        and comment.get("comment") is not None
        and not is_machine_directive_comment(comment["comment"])
        and comment.get("intent") in TARGET_INTENTS
    ]


def generate_comments_for_file(file_data: dict, model_profile: ModelProfile) -> None:
    target_comments = _target_comments(file_data)
    if not target_comments:
        file_data["comment_generations"] = []
        return

    comment_generations = []
    for comment_data in target_comments:
        try:
            comment_generations.append(
                _comment_generation(file_data, comment_data, model_profile)
            )
        except Exception as error:
            logging.warning(
                "Skipping a comment in %s: could not build generation inputs: %s",
                file_data.get("new_path"),
                error,
            )

    file_data["comment_generations"] = comment_generations


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
    run_dir: Path, label: str | None = None, limit: int | None = None
) -> Path:
    """Generate comments for the run's source dataset into a named generation
    subdirectory. Each generation is written independently, so re-running never
    touches a sibling generation. Re-running an existing label overwrites its
    output but leaves its review notes in place (notes are keyed by target).
    Returns the generation directory."""
    files_data = iter_from_jsonl(run_dir, SOURCE_FILENAME)

    model_profile = get_model_profile()
    model_profile_name = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)

    label = label or generations.default_label()
    gen_dir = generations.generation_dir(run_dir, label)
    if (gen_dir / f"{GENERATED_DATASET_FILENAME}.jsonl").exists():
        logging.warning(
            "Generation %r already exists at %s; overwriting its output "
            "(review notes are preserved).",
            label,
            gen_dir,
        )

    generations.write_manifest(
        gen_dir,
        label=label,
        model_profile=model_profile_name,
        model_names=model_profile.model_names,
        config={"max_generate": limit},
    )

    save_to_jsonl([], gen_dir, GENERATED_DATASET_FILENAME)

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
        generate_comments_for_file(file_data, model_profile)
        files_processed += 1

        append_to_jsonl([file_data], gen_dir, GENERATED_DATASET_FILENAME)

    return gen_dir
