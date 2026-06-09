import ast
import logging
import os
import re
from pathlib import Path

from llms.transformers import get_completion
from storage import append_to_jsonl, load_from_jsonl, save_to_jsonl
from comments import get_comments_from_file, strip_comments_from_file

GENERATED_DATASET_FILENAME = "files_generated"

MODEL_PROFILES = {
    "local": [
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    ],
    "cluster": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
}
DEFAULT_MODEL_PROFILE = "local"

ELIGIBLE_CHANGE_TYPES = {"ADD", "MODIFY"}

GENERATION_ATTEMPTS = 3

PROMPT_PATH = Path(__file__).parent / "prompts" / "incremental.md"
PROMPT_NAME = PROMPT_PATH.name

# Inserted into the prompt only for MODIFY files, where a previous version exists.
_PREVIOUS_SECTION_TEMPLATE = (
    "\n## Previous version of the file (before this change)\n"
    "This is the file before the change, including its existing comments, "
    "for context:\n"
    "<<<{previous_code}>>>\n"
)


def get_generate_models() -> list[str]:
    profile = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MODEL_PROFILE {profile!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[profile]


def _build_prompt(new_code: str, previous_code: str | None) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    if previous_code:
        previous_section = _PREVIOUS_SECTION_TEMPLATE.replace(
            "{previous_code}", previous_code
        )
    else:
        previous_section = ""
    return template.replace("{previous_section}", previous_section).replace(
        "{code_file}", new_code
    )


def _assert_valid_python_syntax(source: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ValueError(
            f"Generated code is not valid Python: line {e.lineno}: {e.msg}"
        ) from e


def _parse_generated_code(llm_response: str) -> str:
    stripped_response = llm_response.strip()
    code_fence_match = re.search(
        r"```(?:python)?\s*\n?(.*?)```", stripped_response, re.DOTALL
    )
    if code_fence_match:
        extracted_code = code_fence_match.group(1).strip("\n") + "\n"
    else:
        extracted_code = (
            stripped_response
            if stripped_response.endswith("\n")
            else stripped_response + "\n"
        )
    _assert_valid_python_syntax(extracted_code)
    return extracted_code


def _significant_code_lines(source: str) -> list[str]:
    return [line.rstrip() for line in source.splitlines() if line.strip()]


def _assert_code_preserved(generated_content: str, expected_code: str) -> None:
    generated_code = strip_comments_from_file(generated_content) or ""
    if _significant_code_lines(generated_code) != _significant_code_lines(
        expected_code
    ):
        raise ValueError("Generated output altered the code, not just comments")


def _generate_once(new_code: str, previous_code: str | None, model_name: str) -> str:
    prompt = _build_prompt(new_code, previous_code)
    llm_response = get_completion(model_name, prompt)
    generated_content = _parse_generated_code(llm_response)
    _assert_code_preserved(generated_content, new_code)
    return generated_content


def _build_generation_record(
    model_name: str, generated_content: str, previous_code: str | None
) -> dict:
    annotated_comments = get_comments_from_file(generated_content, previous_code or "")
    return {
        "model": model_name,
        "prompt": PROMPT_NAME,
        "generated_content": generated_content,
        "generated_comments": annotated_comments,
    }


def generate_comments_with_model(
    new_code: str,
    previous_code: str | None,
    model_name: str,
    filepath: str,
) -> dict | None:
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        try:
            generated_content = _generate_once(new_code, previous_code, model_name)
        except Exception as error:
            logging.warning(
                "Model %s attempt %d/%d failed for %s: %s",
                model_name,
                attempt,
                GENERATION_ATTEMPTS,
                filepath,
                error,
            )
            continue

        return _build_generation_record(model_name, generated_content, previous_code)

    logging.warning(
        "Skipping model %s for %s after %d failed attempts",
        model_name,
        filepath,
        GENERATION_ATTEMPTS,
    )
    return None


def _source_file_path(source_file: dict) -> str:
    return source_file.get("new_path") or source_file["filename"]


def generate_comments_for_file(
    source_file: dict, model_names: list[str]
) -> tuple[int, int]:
    new_code = source_file["source_code_without_comments"]
    previous_code = source_file.get("previous_source_code")
    filepath = _source_file_path(source_file)

    successful_generations = []
    failed_model_count = 0

    for model_name in model_names:
        generation = generate_comments_with_model(
            new_code, previous_code, model_name, filepath
        )
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
    all_files = load_from_jsonl(run_dir, "repo_files")
    eligible_files = [
        source_file for source_file in all_files if _is_eligible(source_file)
    ]
    files_to_process = eligible_files[:limit]
    total_files = len(files_to_process)
    model_names = get_generate_models()
    logging.info(
        "Generating comments for %d/%d eligible files across %d models...",
        total_files,
        len(all_files),
        len(model_names),
    )

    save_to_jsonl([], run_dir, GENERATED_DATASET_FILENAME)

    succeeded_generation_count = 0
    failed_generation_count = 0

    for index, source_file in enumerate(files_to_process, start=1):
        filepath = _source_file_path(source_file)
        logging.info(
            "Generating comments for %s (%d/%d)...", filepath, index, total_files
        )
        succeeded_for_file, failed_for_file = generate_comments_for_file(
            source_file, model_names
        )
        succeeded_generation_count += succeeded_for_file
        failed_generation_count += failed_for_file
        append_to_jsonl([source_file], run_dir, GENERATED_DATASET_FILENAME)

    logging.info(
        "Saved enriched dataset to %s (%d generations succeeded, %d failed)",
        run_dir / f"{GENERATED_DATASET_FILENAME}.jsonl",
        succeeded_generation_count,
        failed_generation_count,
    )
