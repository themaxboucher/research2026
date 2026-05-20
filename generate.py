import ast
import logging
import re
from pathlib import Path

from llms.open_router import get_completion
from storage import load_from_json, save_to_json
from comments import get_comments_from_file

GENERATED_DATASET_FILENAME = "files_generated"
GENERATE_MODEL = "meta-llama/llama-3.1-8b-instruct"
PROVIDER = { "only": ["nebius/fp8", "groq"], "allow_fallbacks": False} # only use the providers with the max output tokens
ZERO_SHOT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zero_shot.md"

FILE_GENERATION_LIMIT = 10

def _load_zero_shot_prompt(code_file: str) -> str:
    template = ZERO_SHOT_PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{code_file}", code_file)


def _is_valid_python(source: str) -> tuple[bool, str | None]:
    try:
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def _parse_generated_code(response: str) -> str:
    text = response.strip()
    fence_match = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        code = fence_match.group(1).strip("\n") + "\n"
    else:
        code = text if text.endswith("\n") else text + "\n"
    print(code)
    valid, error = _is_valid_python(code)
    if not valid:
        raise ValueError(f"Generated code is not valid Python: {error}")

    return code

def generate_comments(file: dict) -> None:
    prompt = _load_zero_shot_prompt(file["content_without_comments"])
    response = get_completion(GENERATE_MODEL, prompt, PROVIDER)
    generated_content = _parse_generated_code(response)

    file["generated_content"] = generated_content
    file["generated_comments"] = [comment for comment in get_comments_from_file(generated_content)]
    file["generation"] = {
        "model": GENERATE_MODEL,
        "prompt": ZERO_SHOT_PROMPT_PATH.name,
    }


def generate_comments_for_dataset() -> None:
    dataset = load_from_json("files")
    files_to_process = dataset[:FILE_GENERATION_LIMIT]
    total = len(files_to_process)
    logging.info("Generating comments for %d files...", total)

    succeeded = 0
    skipped = 0

    for index, file in enumerate(files_to_process, start=1):
        logging.info(
            "Generating comments for %s (%d/%d)...",
            file["filepath"],
            index,
            total,
        )
        try:
            generate_comments(file)
            succeeded += 1
        except ValueError as e:
            logging.warning("Skipping %s: %s", file["filepath"], e)
            skipped += 1

    save_to_json(dataset, GENERATED_DATASET_FILENAME)
    logging.info(
        "Saved enriched dataset to data/%s.json (%d succeeded, %d skipped)",
        GENERATED_DATASET_FILENAME,
        succeeded,
        skipped,
    )

if __name__ == "__main__":
    generate_comments_for_dataset()
