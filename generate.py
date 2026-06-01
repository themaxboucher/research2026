import ast
import logging
import re
from pathlib import Path

from llms.transformers import get_completion
from storage import load_from_json, save_to_json
from comments import get_comments_from_file

GENERATED_DATASET_FILENAME = "files_generated"
GENERATE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ZERO_SHOT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zero_shot.md"

FILE_GENERATION_LIMIT = 1


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


def generate_comments(source_file: dict) -> None:
    prompt = _load_zero_shot_prompt(source_file["content_without_comments"])
    llm_response = get_completion(GENERATE_MODEL, prompt)
    generated_content = _parse_generated_code(llm_response)

    source_file["generated_content"] = generated_content
    source_file["generated_comments"] = get_comments_from_file(generated_content)
    source_file["generation"] = {
        "model": GENERATE_MODEL,
        "prompt": ZERO_SHOT_PROMPT_PATH.name,
    }


def generate_comments_for_dataset() -> None:
    all_files = load_from_json("files")
    files_to_process = all_files[:FILE_GENERATION_LIMIT]
    total_files = len(files_to_process)
    logging.info("Generating comments for %d files...", total_files)

    succeeded_count = 0
    skipped_count = 0

    for index, source_file in enumerate(files_to_process, start=1):
        logging.info(
            "Generating comments for %s (%d/%d)...",
            source_file["filepath"],
            index,
            total_files,
        )
        try:
            generate_comments(source_file)
            succeeded_count += 1
        except ValueError as error:
            logging.warning("Skipping %s: %s", source_file["filepath"], error)
            skipped_count += 1

    save_to_json(all_files, GENERATED_DATASET_FILENAME)
    logging.info(
        "Saved enriched dataset to data/%s.json (%d succeeded, %d skipped)",
        GENERATED_DATASET_FILENAME,
        succeeded_count,
        skipped_count,
    )


if __name__ == "__main__":
    generate_comments_for_dataset()
