import io
import json
import logging
import re
import tokenize
from pathlib import Path

from llms.open_router import get_completion
from storage import load_from_json, save_to_json

ANALYSED_DATASET_FILENAME = "files_analysed"

CLASSIFY_COMMENT_MODEL = "meta-llama/llama-3.1-8b-instruct"
CLASSIFY_COMMENT_PROMPT_PATH = Path(__file__).parent / "prompts" / "classify_comment.md"
CODE_CONTEXT_LINES = 10

FILE_ANALYSIS_LIMIT = 5

def count_loc(file_content: str) -> int:
    return len(file_content.splitlines())

def count_comments(file_content: str) -> int:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    return len([tok for tok in tokens if tok.type == tokenize.COMMENT])

def avg_comment_character_length(file_content: str) -> float:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    if len([tok for tok in tokens if tok.type == tokenize.COMMENT]) == 0:
        return 0
    return sum([len(tok.string.rstrip("\r\n")) for tok in tokens if tok.type == tokenize.COMMENT]) / len([tok for tok in tokens if tok.type == tokenize.COMMENT])

def get_code_context(file_content: str, start_line: int, end_line: int, context_lines: int = CODE_CONTEXT_LINES) -> str:
    lines = file_content.splitlines()
    context_start = max(0, start_line - 1 - context_lines)
    context_end = min(len(lines), end_line + context_lines)
    return "\n".join(lines[context_start:context_end])

def _load_classify_prompt(comment_text: str, code_context: str) -> str:
    template = CLASSIFY_COMMENT_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{comment_text}", comment_text)
        .replace("{code_context}", code_context)
    )

def _parse_classification_response(response: str) -> dict:
    text = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid classification JSON: {e}") from e

def classify_comment(comment: dict, file_content: str) -> dict:
    code_context = get_code_context(file_content, comment["start_line"], comment["end_line"])
    prompt = _load_classify_prompt(comment["comment"], code_context)
    response = get_completion(CLASSIFY_COMMENT_MODEL, prompt)
    return _parse_classification_response(response)

def analyse_file_content(file_content: str) -> dict:
    loc = count_loc(file_content)
    comments = count_comments(file_content)
    comments_per_loc = comments / loc if loc else 0
    avg_comment_length = avg_comment_character_length(file_content)

    return {
        "loc": loc,
        "comments": comments,
        "comments_per_loc": comments_per_loc,
        "avg_comment_length": avg_comment_length,
    }

def _enrich_comment(comment: dict, file_content: str) -> None:
    classification = classify_comment(comment, file_content)
    comment["classification"] = {
        **classification,
        "model": CLASSIFY_COMMENT_MODEL,
        "prompt": CLASSIFY_COMMENT_PROMPT_PATH.name,
    }

def analyse_file_record(file: dict) -> None:
    if file.get("content"):
        file["original_metrics"] = analyse_file_content(file["content"])

    if file.get("generated_content"):
        file["generated_metrics"] = analyse_file_content(file["generated_content"])

    if file.get("comments"):
        for comment in file["comments"]:
            _enrich_comment(comment, file["content"])
            
    if file.get("generated_comments"):
        for comment in file["generated_comments"]:
            _enrich_comment(comment, file["generated_content"])

def analyse_dataset() -> None:
    dataset = load_from_json("files_generated")
    files_to_process = dataset[:FILE_ANALYSIS_LIMIT]
    total = len(files_to_process)
    logging.info("Analysing %d files...", total)

    succeeded = 0
    skipped = 0

    for index, file in enumerate(files_to_process, start=1):
        logging.info(
            "Analysing %s (%d/%d)...",
            file["filepath"],
            index,
            total,
        )
        try:
            analyse_file_record(file)
            succeeded += 1
        except ValueError as e:
            logging.warning("Skipping %s: %s", file["filepath"], e)
            skipped += 1

    save_to_json(dataset, ANALYSED_DATASET_FILENAME)
    logging.info(
        "Saved enriched dataset to data/%s.json (%d succeeded, %d skipped)",
        ANALYSED_DATASET_FILENAME,
        succeeded,
        skipped,
    )

if __name__ == "__main__":
    analyse_dataset()