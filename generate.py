import ast
import difflib
import logging
import os
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
            "openai/gpt-5-codex",
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

# Lines of context on each side of a module-level comment with no enclosing
# function or class.
MODULE_SCOPE_WINDOW = 15

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
    enclosing_scopes = []
    for node, qualified_name in _ast_nodes(source):
        start, end = _node_line_span(node)
        if start <= line <= end:
            span_length = end - start
            enclosing_scopes.append((span_length, qualified_name))
    if not enclosing_scopes:
        return None
    _, innermost_name = min(enclosing_scopes, key=lambda scope: scope[0])
    return innermost_name


def _scope_code(source_code: str, qualified_name: str) -> str | None:
    source_lines = source_code.splitlines()
    for node, name in _ast_nodes(source_code):
        if name != qualified_name:
            continue

        start, end = _node_line_span(node)

        # Include a block comment if one exists for above the scope
        while start > 1 and source_lines[start - 2].strip().startswith("#"):
            start -= 1

        return "\n".join(source_lines[start - 1 : end])
    return None


def _create_diff(old_code: str, new_code: str, filepath: str) -> str:
    diff_lines = difflib.unified_diff(
        (old_code or "").splitlines(keepends=True),
        (new_code or "").splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff_lines)


def _anchored_line_no(target: dict, source_lines: list[str]) -> int:
    if target["type"] == "inline":
        return target["start_line"]
    for line_no in range(target["end_line"] + 1, len(source_lines) + 1):
        stripped = source_lines[line_no - 1].strip()
        line_is_code = stripped and not stripped.startswith("#")
        if line_is_code:
            return line_no
    return target["end_line"]


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
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not first.startswith("#"):
        first = "# " + first.lstrip("#").strip()
    return first


def code_line_indentation(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _replace_inline_comment(line: str, target: dict, comment_text: str) -> str:
    code_part = target.get("anchor") or line.split("#", 1)[0].rstrip()
    return code_part + "  " + _normalize_inline_comment(comment_text)


def _apply_comment(source_code: str, target: dict, comment_text: str) -> str:
    lines = source_code.split("\n")
    start_index = target["start_line"] - 1

    if target["type"] == "inline":
        lines[start_index] = _replace_inline_comment(
            lines[start_index], target, comment_text
        )
        return "\n".join(lines)

    indentation = code_line_indentation(lines[start_index])
    block_lines = [
        indentation + comment_line
        for comment_line in _normalize_block_comment(comment_text)
    ]
    lines[start_index : target["end_line"]] = block_lines
    return "\n".join(lines)


class CommentTask(NamedTuple):
    filepath: str
    comment_data: dict
    scope_code: str
    scope_diff: str
    reverted_comment: str | None
    source_code: str
    scope_qualified_name: str | None


def _build_comment_task(
    file_data: dict, comment_data: dict, old_comments: list[dict]
) -> CommentTask:
    source_code = file_data["source_code"]
    previous_source_code = file_data["previous_source_code"]
    filepath = file_data["new_path"]

    anchored_line_no = _anchored_line_no(comment_data, source_code.splitlines())
    qualified_name = _enclosing_scope_name(source_code, anchored_line_no)

    scope_is_module = qualified_name is None

    if scope_is_module:
        scope_diff = _create_diff(previous_source_code, source_code, filepath)
    else:
        new_scope_code = _scope_code(source_code, qualified_name)
        old_scope_code = _scope_code(previous_source_code, qualified_name)

        scope_diff = _create_diff(old_scope_code, new_scope_code, filepath)

    reverted_comment = None
    if comment_data["status"] == "modified":
        reverted_comment = _reverted_comment_text(comment_data, old_comments)

    return CommentTask(
        filepath=filepath,
        comment_data=comment_data,
        scope_code=new_scope_code,
        scope_diff=scope_diff,
        reverted_comment=reverted_comment,
        source_code=source_code,
        scope_qualified_name=qualified_name,
    )


def _location_instruction(task: CommentTask) -> str:
    anchor = task.comment_data.get("anchor") or "(the anchored code)"
    is_comment_edit = task.comment_data["status"] == "modified"
    is_inline_comment = task.comment_data["type"] == "inline"

    if is_inline_comment and is_comment_edit:
        return (
            "Update the inline comment on this line of code:\n"
            f"    {anchor}\n"
            f"Current (outdated) comment: {task.reverted_comment}"
        )

    if is_inline_comment:
        return f"Write a new inline comment for this line of code:\n    {anchor}"

    if is_comment_edit:
        return (
            "Update the block comment directly above this line of code:\n"
            f"    {anchor}\n"
            "Current (outdated) comment:\n"
            f"{task.reverted_comment}"
        )

    return f"Write a new block comment directly above this line of code:\n    {anchor}"


def _build_prompt(task: CommentTask) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{file_path}", task.filepath)
        .replace("{comment_type}", task.comment_data["type"])
        .replace("{scope_code}", task.scope_code)
        .replace("{scope_diff}", task.scope_diff)
        .replace("{location_instruction}", _location_instruction(task))
    )


def _generate_comment_with_model(
    task: CommentTask,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict:
    prompt = _build_prompt(task)
    raw_response = get_completion(model_name, prompt)
    try:
        comment_text = raw_response.strip()
        if not comment_text:
            raise ValueError("Model returned an empty comment")
        applied_code = _apply_comment(task.source_code, task.comment_data, comment_text)
        ast.parse(applied_code)
    except Exception as error:
        logging.warning(
            "Failed to generate comment in %s with model %s: %s",
            task.filepath,
            model_name,
            error,
        )
        return {
            "model": model_name,
            "prompt": prompt,
            "raw_response": raw_response,
            "comment": None,
            "applied_code": None,
            "error": str(error),
        }
    return {
        "model": model_name,
        "prompt": prompt,
        "raw_response": raw_response,
        "comment": comment_text,
        "applied_code": applied_code,
        "error": None,
    }


def _run_all_models(task: CommentTask, model_profile: ModelProfile) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(model_profile.model_names)) as executor:
        futures = [
            executor.submit(
                _generate_comment_with_model,
                task,
                model_name,
                model_profile.get_completion,
            )
            for model_name in model_profile.model_names
        ]
        return [future.result() for future in as_completed(futures)]


def _target_comments(source_file: dict) -> list[dict]:
    TARGET_COMMENT_TYPES = {"inline", "block"}
    TARGET_COMMENT_STATUSES = {"added", "modified"}
    return [
        comment
        for comment in (source_file.get("comments") or [])
        if comment.get("type") in TARGET_COMMENT_TYPES
        and comment.get("status") in TARGET_COMMENT_STATUSES
    ]


def _comment_generation(
    file_data: dict,
    comment_data: dict,
    old_comments: list[dict],
    model_profile: ModelProfile,
) -> dict:
    task = _build_comment_task(file_data, comment_data, old_comments)
    results = _run_all_models(task, model_profile)
    return {
        "type": comment_data["type"],
        "status": comment_data["status"],
        "start_line": comment_data["start_line"],
        "end_line": comment_data["end_line"],
        "anchor": comment_data.get("anchor"),
        "comment": comment_data.get("comment"),
        "reverted_comment": task.reverted_comment,
        "scope_qualified_name": task.scope_qualified_name,
        "scope_code": task.scope_code,
        "scope_diff": task.scope_diff,
        "results": results,
    }


def generate_comments_for_file(file_data: dict, model_profile: ModelProfile) -> None:
    target_comments = _target_comments(file_data)
    if not target_comments:
        file_data["comment_generations"] = []
        return

    previous_comments = extract_comments(
        file_data.get("previous_source_code"), include_docstrings=False
    )

    comment_generations = []
    for comment_data in target_comments:
        try:
            comment_generations.append(
                _comment_generation(
                    file_data, comment_data, previous_comments, model_profile
                )
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
