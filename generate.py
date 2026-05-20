import ast
import io
import logging
import re
import tokenize
from pathlib import Path

from llms.open_router import get_completion
from storage import load_from_json, save_to_json

GENERATED_DATASET_FILENAME = "files_generated"
GENERATE_MODEL = "meta-llama/llama-3.1-8b-instruct"
PROVIDER = { "only": ["nebius/fp8", "groq"], "allow_fallbacks": False} # only use the providers with the max output tokens
ZERO_SHOT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zero_shot.md"

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


def _comments_by_line(file_content: str) -> dict[int, list[str]]:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    by_line: dict[int, list[str]] = {}
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        line_no = tok.start[0]
        by_line.setdefault(line_no, []).append(tok.string.rstrip("\r\n"))
    return by_line


def _extract_new_comments(
    filepath: str,
    generated_content: str,
    base_content: str,
) -> list[dict]:
    base_by_line = _comments_by_line(base_content)
    tokens = list(tokenize.tokenize(io.BytesIO(generated_content.encode("utf-8")).readline))
    lines = generated_content.splitlines(keepends=True)

    found_comments: list[tuple[int, str, bool]] = []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        comment_text = tok.string.rstrip("\r\n")
        if comment_text in base_by_line.get(row, []):
            continue
        line_text = lines[row - 1] if 0 < row <= len(lines) else ""
        prefix = line_text[:col]
        is_standalone = prefix.strip() == ""
        found_comments.append((row, comment_text, is_standalone))

    if not found_comments:
        return []

    found_comments.sort(key=lambda x: x[0])

    comment_blocks: list[dict] = []
    for line_no, text, is_standalone in found_comments:
        if (
            comment_blocks
            and is_standalone
            and comment_blocks[-1]["_standalone"]
            and comment_blocks[-1]["end_line"] + 1 == line_no
        ):
            comment_blocks[-1]["comment"] += "\n" + text
            comment_blocks[-1]["end_line"] = line_no
        else:
            comment_blocks.append({
                "filepath": filepath,
                "comment": text,
                "start_line": line_no,
                "end_line": line_no,
                "_standalone": is_standalone,
            })

    for block in comment_blocks:
        block.pop("_standalone", None)

    return comment_blocks


def generate_comments(file: dict) -> None:
    prompt = _load_zero_shot_prompt(file["content_without_comments"])
    response = get_completion(GENERATE_MODEL, prompt, PROVIDER)
    generated_content = _parse_generated_code(response)

    file["generated_content"] = generated_content
    file["generated_comments"] = [
        {
            **comment,
            "repo_owner": file["repo_owner"],
            "repo_name": file["repo_name"],
            "sha": file["sha"],
        }
        for comment in _extract_new_comments(
            file["filepath"],
            generated_content,
            file["content_without_comments"],
        )
    ]
    file["generation"] = {
        "model": GENERATE_MODEL,
        "prompt": ZERO_SHOT_PROMPT_PATH.name,
    }


def generate_comments_for_dataset() -> None:
    dataset = load_from_json("files")
    logging.info("Generating comments for %d files...", len(dataset))

    for index, file in enumerate(dataset, start=1):
        logging.info(
            "Generating comments for %s (%d/%d)...",
            file["filepath"],
            index,
            len(dataset),
        )
        generate_comments(file)

    save_to_json(dataset, GENERATED_DATASET_FILENAME)
    logging.info("Saved enriched dataset to data/%s.json", GENERATED_DATASET_FILENAME)

if __name__ == "__main__":
    generate_comments_for_dataset()
