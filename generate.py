import ast
import difflib
import logging
import os
from pathlib import Path
from typing import Callable, NamedTuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llms import openrouter, transformers
from storage import append_to_jsonl, iter_from_jsonl, save_to_jsonl
from comments import (
    get_comments_from_file,
    graft_comments_onto_code,
    strip_comments_from_file,
)
from edits import SearchReplaceEdit, apply_edits, parse_edit_response


GENERATED_DATASET_FILENAME = "files_generated"


class ModelProfile(NamedTuple):
    model_names: list[str]
    get_completion: Callable[[str, str], str]


MODEL_PROFILES = {
    "local": ModelProfile(
        model_names=[
            "meta-llama/llama-3.1-8b-instruct",
            "qwen/qwen-2.5-7b-instruct",
            "openai/gpt-5.5",
            "anthropic/claude-sonnet-4.6",
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

ELIGIBLE_CHANGE_TYPES = {"ADD", "MODIFY"}

GENERATION_ATTEMPTS = 3

PROMPT_PATH = Path(__file__).parent / "prompts" / "file_few_shot.md"
PROMPT_NAME = PROMPT_PATH.name


class GenerationTask(NamedTuple):
    filepath: str
    code_with_outdated_comments: str
    comment_free_diff: str
    previous_code: str | None


def get_model_profile() -> ModelProfile:
    profile = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MODEL_PROFILE {profile!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[profile]


def _build_prompt(task: GenerationTask) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{file_path}", task.filepath)
        .replace("{code_file}", task.code_with_outdated_comments)
        .replace("{diff}", task.comment_free_diff)
    )


def _build_comment_free_diff(
    previous_code_without_comments: str | None,
    code_without_comments: str,
    filepath: str,
) -> str:
    diff_lines = difflib.unified_diff(
        (previous_code_without_comments or "").splitlines(keepends=True),
        code_without_comments.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff_lines)


def _assert_valid_python_syntax(source: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ValueError(
            f"Generated code is not valid Python: line {e.lineno}: {e.msg}"
        ) from e


def _significant_code_lines(source: str) -> list[str]:
    return [line.rstrip() for line in source.splitlines() if line.strip()]


def _assert_code_preserved(generated_content: str, original_content: str) -> None:
    generated_code = strip_comments_from_file(generated_content) or ""
    original_code = strip_comments_from_file(original_content) or ""
    if _significant_code_lines(generated_code) != _significant_code_lines(
        original_code
    ):
        raise ValueError("Generated output altered the code, not just comments")


def _generate_once(
    task: GenerationTask,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> tuple[str, list[SearchReplaceEdit]]:
    prompt = _build_prompt(task)
    llm_response = get_completion(model_name, prompt)
    try:
        edits = parse_edit_response(llm_response)
        generated_content = apply_edits(task.code_with_outdated_comments, edits)
        _assert_valid_python_syntax(generated_content)
        _assert_code_preserved(generated_content, task.code_with_outdated_comments)
    except Exception:
        logging.warning(
            "Raw response from %s for %s:\n%s",
            model_name,
            task.filepath,
            llm_response,
        )
        raise
    return generated_content, edits


def _build_generation_record(
    model_name: str,
    generated_content: str,
    edits: list[SearchReplaceEdit],
    previous_code: str | None,
) -> dict:
    annotated_comments = get_comments_from_file(generated_content, previous_code or "")
    return {
        "model": model_name,
        "prompt": PROMPT_NAME,
        "generated_content": generated_content,
        "generated_edits": [edit._asdict() for edit in edits],
        "generated_comments": annotated_comments,
    }


def generate_comments_with_model(
    task: GenerationTask,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict | None:
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            generated_content, edits = _generate_once(
                task, model_name, get_completion
            )
        except Exception as error:
            logging.warning(
                "Model %s attempt %d/%d failed for %s: %s",
                model_name,
                attempt,
                GENERATION_ATTEMPTS,
                task.filepath,
                error,
            )
            continue

        return _build_generation_record(
            model_name, generated_content, edits, task.previous_code
        )

    logging.warning(
        "Skipping model %s for %s after %d failed attempts",
        model_name,
        task.filepath,
        GENERATION_ATTEMPTS,
    )
    return None


def _source_file_path(source_file: dict) -> str:
    return source_file.get("new_path") or source_file["filename"]


def _build_generation_task(source_file: dict) -> GenerationTask:
    filepath = _source_file_path(source_file)
    code_without_comments = source_file["source_code_without_comments"]
    previous_code = source_file.get("previous_source_code")
    code_with_outdated_comments = graft_comments_onto_code(
        code_without_comments, previous_code
    )
    comment_free_diff = _build_comment_free_diff(
        source_file.get("previous_source_code_without_comments"),
        code_without_comments,
        filepath,
    )
    return GenerationTask(
        filepath=filepath,
        code_with_outdated_comments=code_with_outdated_comments,
        comment_free_diff=comment_free_diff,
        previous_code=previous_code,
    )


def generate_comments_for_file(
    source_file: dict, model_profile: ModelProfile
) -> tuple[int, int]:
    try:
        task = _build_generation_task(source_file)
    except Exception as error:
        logging.warning(
            "Skipping %s: could not build generation inputs: %s",
            _source_file_path(source_file),
            error,
        )
        source_file["generations"] = []
        return 0, len(model_profile.model_names)

    successful_generations = []
    failed_model_count = 0

    num_models = len(model_profile.model_names)

    with ThreadPoolExecutor(max_workers=num_models) as executor:
        generation_futures = [
            executor.submit(
                generate_comments_with_model,
                task,
                model_name,
                model_profile.get_completion,
            )
            for model_name in model_profile.model_names
        ]

        for generation_future in as_completed(generation_futures):
            generation = generation_future.result()
            if generation is None:
                failed_model_count += 1
            else:
                successful_generations.append(generation)

    source_file["generations"] = successful_generations
    return len(successful_generations), failed_model_count


def _is_eligible(source_file: dict) -> bool:
    if source_file.get("change_type") not in ELIGIBLE_CHANGE_TYPES:
        return False
    return bool(source_file.get("source_code_without_comments"))


def generate_comments_for_dataset(run_dir: Path, limit: int | None = None) -> None:
    file_data = iter_from_jsonl(run_dir, "repo_files")

    model_profile = get_model_profile()

    save_to_jsonl([], run_dir, GENERATED_DATASET_FILENAME)  # Clears the output file

    succeeded_generation_count = 0
    failed_generation_count = 0
    files_processed = 0

    for source_file in file_data:
        if not _is_eligible(source_file):
            continue

        if limit is not None and files_processed >= limit:
            break

        filepath = _source_file_path(source_file)
        logging.info(
            "Generating comments for %s (file %d/%d)",
            filepath,
            files_processed + 1,
            limit,
        )
        succeeded_for_file, failed_for_file = generate_comments_for_file(
            source_file, model_profile
        )
        succeeded_generation_count += succeeded_for_file
        failed_generation_count += failed_for_file
        files_processed += 1
        append_to_jsonl([source_file], run_dir, GENERATED_DATASET_FILENAME)

    logging.info(
        "Saved enriched dataset to %s (%d generations succeeded, %d failed)",
        run_dir / f"{GENERATED_DATASET_FILENAME}.jsonl",
        succeeded_generation_count,
        failed_generation_count,
    )
