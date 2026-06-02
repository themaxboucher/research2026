import ast
import logging
import re
from pathlib import Path

from llms.transformers import get_completion
from runs import require_latest_run_directory
from storage import load_from_json, save_to_json
from comments import get_comments_from_file

GENERATED_DATASET_FILENAME = "files_generated"
GENERATE_MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct", 
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]
ZERO_SHOT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zero_shot.md"


def _load_zero_shot_prompt(source_code: str) -> str:
    template = ZERO_SHOT_PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{code_file}", source_code)


def _assert_valid_python_syntax(source: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Generated code is not valid Python: line {e.lineno}: {e.msg}") from e


def _parse_generated_code(llm_response: str) -> str:
    stripped_response = llm_response.strip()
    code_fence_match = re.search(r"```(?:python)?\s*\n?(.*?)```", stripped_response, re.DOTALL)
    if code_fence_match:
        extracted_code = code_fence_match.group(1).strip("\n") + "\n"
    else:
        extracted_code = stripped_response if stripped_response.endswith("\n") else stripped_response + "\n"
    _assert_valid_python_syntax(extracted_code)
    return extracted_code


def generate_comments_with_model(source_code_without_comments: str, model_name: str) -> dict:
    prompt = _load_zero_shot_prompt(source_code_without_comments)
    llm_response = get_completion(model_name, prompt)
    generated_content = _parse_generated_code(llm_response)
    return {
        "model": model_name,
        "prompt": ZERO_SHOT_PROMPT_PATH.name,
        "generated_content": generated_content,
        "generated_comments": get_comments_from_file(generated_content),
    }


def generate_comments_for_file(source_file: dict) -> tuple[int, int]:
    source_code_without_comments = source_file["content_without_comments"]
    successful_generations = []
    failed_model_count = 0

    for model_name in GENERATE_MODELS:
        try:
            generation = generate_comments_with_model(source_code_without_comments, model_name)
            successful_generations.append(generation)
        except ValueError as error:
            logging.warning(
                "Skipping model %s for %s: %s",
                model_name,
                source_file["filepath"],
                error,
            )
            failed_model_count += 1

    source_file["generations"] = successful_generations
    return len(successful_generations), failed_model_count


def generate_comments_for_dataset(run_dir: Path, limit: int | None = None) -> None:
    all_files = load_from_json(run_dir, "files")
    files_to_process = all_files[:limit]
    total_files = len(files_to_process)
    model_count = len(GENERATE_MODELS)
    logging.info(
        "Generating comments for %d files across %d models...",
        total_files,
        model_count,
    )

    succeeded_generation_count = 0
    failed_generation_count = 0

    for index, source_file in enumerate(files_to_process, start=1):
        logging.info(
            "Generating comments for %s (%d/%d)...",
            source_file["filepath"],
            index,
            total_files,
        )
        succeeded_for_file, failed_for_file = generate_comments_for_file(source_file)
        succeeded_generation_count += succeeded_for_file
        failed_generation_count += failed_for_file

    save_to_json(all_files, run_dir, GENERATED_DATASET_FILENAME)
    logging.info(
        "Saved enriched dataset to %s (%d generations succeeded, %d failed)",
        run_dir / f"{GENERATED_DATASET_FILENAME}.json",
        succeeded_generation_count,
        failed_generation_count,
    )


if __name__ == "__main__":
    generate_comments_for_dataset(require_latest_run_directory())
