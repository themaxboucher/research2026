import ast
import logging
import os
import re
from pathlib import Path
from typing import Callable, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llms import openrouter, transformers
from storage import append_to_jsonl, iter_from_jsonl, save_to_jsonl
from comments import extract_comments


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

PROMPT_PATH = Path(__file__).parent / "prompts" / "comment_local.md"
PROMPT_NAME = PROMPT_PATH.name


def get_model_profile() -> ModelProfile:
    profile = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MODEL_PROFILE {profile!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[profile]


def _ast_nodes(source: str) -> list[tuple[ast.AST, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    nodes: list[tuple[ast.AST, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified_name = prefix + child.name
                nodes.append((child, qualified_name))
                visit(child, qualified_name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return nodes


def _node_line_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    # Include decorator lines in the span, if any
    for decorator in getattr(node, "decorator_list", []):
        start = min(start, decorator.lineno)
    return start, node.end_lineno or node.lineno


def _enclosing_scope_name(source: str, line: int) -> str | None:
    enclosing_scopes = []  # List of scopes enclosing the given line, ordered by span length
    for node, qualified_name in _ast_nodes(source):
        start, end = _node_line_span(node)
        line_is_within_span = start <= line <= end
        if line_is_within_span:
            span_length = end - start
            enclosing_scopes.append((span_length, qualified_name))
    if not enclosing_scopes:
        return None
    _, innermost_name = min(enclosing_scopes, key=lambda scope: scope[0])
    return innermost_name


def _local_scope_bounds(
    source_lines: list[str], source_code: str, qualified_name: str
) -> tuple[int, int] | None:
    for node, name in _ast_nodes(source_code):
        if name != qualified_name:
            continue

        # _node_line_span already extends the start up to cover decorators.
        start, end = _node_line_span(node)

        # Pull in any block comment lines sitting directly above the scope.
        while start > 1 and source_lines[start - 2].strip().startswith("#"):
            start -= 1

        return start, end
    return None


def _scope_code(source_code: str, anchor_line_no: int | None) -> str:
    MAX_LINE_COUNT = 500

    if anchor_line_no is None:
        return "\n".join(source_code.splitlines()[:MAX_LINE_COUNT])

    source_lines = source_code.splitlines()

    qualified_name = _enclosing_scope_name(source_code, anchor_line_no)

    if qualified_name is None:
        top = anchor_line_no
        bottom = anchor_line_no

        while (bottom - top + 1) < min(MAX_LINE_COUNT, len(source_lines)):
            if top > 1:
                top -= 1
            if bottom < len(source_lines):
                bottom += 1

        return "\n".join(source_lines[top - 1 : bottom])

    scope_bounds = _local_scope_bounds(source_lines, source_code, qualified_name)
    if scope_bounds is not None:
        scope_start, scope_end = scope_bounds
        return "\n".join(source_lines[scope_start - 1 : scope_end])

    return "\n".join(source_code.splitlines()[:MAX_LINE_COUNT])  # Fallback


_HUNK_HEADER_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _relevant_hunk(diff: str, target_line: int) -> str:
    if not diff:
        return ""

    hunks: list[tuple[int, int, list[str]]] = []
    current_lines: list[str] = []
    current_new_start = 0
    current_new_count = 0

    for line in diff.splitlines(keepends=True):
        header_match = _HUNK_HEADER_PATTERN.match(line)
        if header_match:
            if current_lines:
                hunks.append((current_new_start, current_new_count, current_lines))
            current_new_start = int(header_match.group(1))
            current_new_count = (
                int(header_match.group(2)) if header_match.group(2) else 1
            )
            current_lines = [line]
        elif current_lines:
            current_lines.append(line)

    if current_lines:
        hunks.append((current_new_start, current_new_count, current_lines))

    for new_start, new_count, hunk_lines in hunks:
        new_end = new_start + max(new_count - 1, 0)
        if new_start <= target_line <= new_end:
            return "".join(hunk_lines)

    return ""


def _anchor_line_no(comment_data: dict, source_code: str) -> int | None:
    if comment_data["type"] == "inline":
        return comment_data["start_line"]

    anchor = comment_data.get("anchor")
    if not anchor:
        return None

    # We can't use the comment data's end_line because it will have moved after
    # the comment was removed/reverted, so we re-locate the anchor by text. A
    # block comment always sits above its anchor, so only consider matches at or
    # below the comment's original start_line and pick the closest one.
    original_line = comment_data["start_line"]
    matching_line_nos = [
        index + 1
        for index, line in enumerate(source_code.splitlines())
        if line.strip() == anchor and index + 1 >= original_line
    ]
    if not matching_line_nos:
        return None
    return min(matching_line_nos)


def _reverted_comment_text(target: dict, old_comments: list[dict]) -> str | None:
    for old_comment in old_comments:
        if (
            old_comment["type"] == target["type"]
            and old_comment.get("anchor") == target.get("anchor")
            and old_comment.get("comment") != target.get("comment")
        ):
            return old_comment.get("comment")
    return None


def _normalize_block_comment(text: str) -> list[str]:
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line.startswith("#")]
    return lines or ["# " + text.strip().lstrip("#").strip()]


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


def _location_instruction(comment_data: dict, unmodified_comment: str | None) -> str:
    anchor = comment_data.get("anchor") or "(the anchored code)"
    is_comment_edit = comment_data["status"] == "modified"
    is_inline_comment = comment_data["type"] == "inline"

    if is_inline_comment and is_comment_edit:
        return (
            "Update the inline comment on this line of code:\n"
            f"    {anchor}\n"
            f"Current (outdated) comment: {unmodified_comment}"
        )

    if is_inline_comment:
        return f"Write a new inline comment for this line of code:\n    {anchor}"

    if is_comment_edit:
        return (
            "Update the block comment directly above this line of code:\n"
            f"    {anchor}\n"
            "Current (outdated) comment:\n"
            f"{unmodified_comment}"
        )

    return f"Write a new block comment directly above this line of code:\n    {anchor}"


def _build_prompt(
    filepath: str,
    comment_data: dict,
    scope_code: str,
    diff_hunk: str,
    unmodified_comment: str | None,
) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{file_path}", filepath)
        .replace("{comment_type}", comment_data["type"])
        .replace("{scope_code}", scope_code)
        .replace("{diff_hunk}", diff_hunk)
        .replace(
            "{location_instruction}",
            _location_instruction(comment_data, unmodified_comment),
        )
    )


def _strip_inline_comment(line: str, anchor: str | None) -> str:
    if anchor is not None:
        return anchor
    naive_anchor_fallback = line.split("#", 1)[0].rstrip()
    return naive_anchor_fallback


def _strip_target_comment(
    source_code: str, comment_data: dict, unmodified_comment: str | None
) -> str:
    is_modified_comment = comment_data["status"] == "modified"
    if is_modified_comment and unmodified_comment is not None:
        # Restore the original comment so the model never sees the human's edit.
        return _apply_new_comment(source_code, comment_data, unmodified_comment)

    source_code_lines = source_code.splitlines()
    start_index = comment_data["start_line"] - 1

    is_inline_comment = comment_data["type"] == "inline"
    if is_inline_comment:
        source_code_lines[start_index] = _strip_inline_comment(
            source_code_lines[start_index], comment_data.get("anchor")
        )
    else:
        del source_code_lines[start_index : comment_data["end_line"]]

    return "\n".join(source_code_lines)


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
        comment_text = raw_response.strip()
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
            "prompt": prompt,
            "raw_response": raw_response,
            "comment_text": None,
            "new_source_code": None,
            "error": str(error),
        }
    return {
        "model": model_name,
        "prompt": prompt,
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

    unmodified_comment = None
    if comment_data["status"] == "modified":
        previous_comments = extract_comments(
            previous_source_code, include_docstrings=False
        )
        unmodified_comment = _reverted_comment_text(comment_data, previous_comments)

    source_code_without_target_comment = _strip_target_comment(
        source_code, comment_data, unmodified_comment
    )

    anchor_line_no = _anchor_line_no(comment_data, source_code_without_target_comment)

    scope_code = _scope_code(source_code_without_target_comment, anchor_line_no)

    diff_hunk = _relevant_hunk(diff, comment_data["start_line"])

    prompt = _build_prompt(
        filepath, comment_data, scope_code, diff_hunk, unmodified_comment
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
        "unmodified_comment": unmodified_comment,
        "scope_code": scope_code,
        "diff_hunk": diff_hunk,
        "results": results,
    }


def _target_comments(source_file: dict) -> list[dict]:
    TARGET_COMMENT_TYPES = {"inline", "block"}
    TARGET_COMMENT_STATUSES = {"added", "modified"}
    return [
        comment
        for comment in (source_file.get("comments") or [])
        if comment.get("type") in TARGET_COMMENT_TYPES
        and comment.get("status") in TARGET_COMMENT_STATUSES
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

    return True


def generate_comments_for_dataset(run_dir: Path, limit: int | None = None) -> None:
    files_data = iter_from_jsonl(run_dir, SOURCE_FILENAME)

    model_profile = get_model_profile()

    save_to_jsonl([], run_dir, GENERATED_DATASET_FILENAME)  # Clears the output file

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

        append_to_jsonl([file_data], run_dir, GENERATED_DATASET_FILENAME)
