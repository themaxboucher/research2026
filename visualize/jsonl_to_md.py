import argparse
import json
import re
from pathlib import Path

MAX_HEADING_LEVEL = 6

_CODE_TOKENS = (
    "()", "[]", "{}", "->", "=>", "::", "'''", '"""',
    "def ", "class ", "import ", "assert", "return ", " = ", "==", "!=",
)
_MARKDOWN_LINE_STARTS = set("#>-*+|=`")


def _is_scalar(value) -> bool:
    return not isinstance(value, (dict, list))


def _scalar_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _looks_like_code(line: str) -> bool:
    stripped = line.lstrip()
    if stripped[:1] in _MARKDOWN_LINE_STARTS:
        return True
    if any(token in line for token in _CODE_TOKENS):
        return True
    if " " not in line and ("/" in line or ("." in line and not line.replace(".", "").isdigit())):
        return True
    return False


def _fence(text: str, language: str = "") -> str:
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest_run + 1)
    return f"{ticks}{language}\n{text}\n{ticks}\n\n"


def _inline_code(text: str) -> str:
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * (longest_run + 1)
    padding = " " if longest_run else ""
    return f"{ticks}{padding}{text}{padding}{ticks}"


def _blockquote(text: str) -> str:
    lines = [f"> {line}" if line else ">" for line in text.split("\n")]
    return "\n".join(lines) + "\n\n"


def _string_to_md(value: str) -> str:
    if value == "":
        return '""\n\n'
    if "\n" in value:
        if "```" in value:
            return _blockquote(value)
        return _fence(value)
    if _looks_like_code(value):
        return f"{_inline_code(value)}\n\n"
    return f"{value}\n\n"


def _heading(level: int, text: str) -> str:
    if level <= MAX_HEADING_LEVEL:
        return f"{'#' * level} {text}\n\n"
    return f"**{text}**\n\n"


def _table_cell(value) -> str:
    return _scalar_text(value).replace("|", "\\|").replace("\n", "<br>")


def _can_render_as_table(record: dict) -> bool:
    if not record:
        return False
    for value in record.values():
        if not _is_scalar(value):
            return False
        if isinstance(value, str) and "\n" in value:
            return False
    return True


def _table_to_md(record: dict) -> str:
    rows = ["| Key | Value |", "| --- | --- |"]
    for key, value in record.items():
        rows.append(f"| {_table_cell(key)} | {_table_cell(value)} |")
    return "\n".join(rows) + "\n\n"


def _scalar_value_to_md(value) -> str:
    if isinstance(value, str):
        return _string_to_md(value)
    return f"{_scalar_text(value)}\n\n"


def _list_to_md(values: list, level: int, label: str) -> str:
    if not values:
        return "[]\n\n"
    if all(_is_scalar(item) for item in values):
        lines = []
        for item in values:
            if isinstance(item, str) and _looks_like_code(item) and "\n" not in item:
                lines.append(f"- {_inline_code(item)}")
            else:
                lines.append(f"- {_scalar_text(item)}")
        return "\n".join(lines) + "\n\n"

    md = ""
    for index, item in enumerate(values, start=1):
        if index > 1:
            md += "---\n\n"
        md += _heading(level, f"{label} {index}")
        md += _value_to_md(item, level + 1, f"{label} {index}")
    return md


def _record_to_md(record: dict, level: int) -> str:
    if not record:
        return "{}\n\n"
    if _can_render_as_table(record):
        return _table_to_md(record)
    md = ""
    for key, value in record.items():
        md += _heading(level, key)
        md += _value_to_md(value, level + 1, key)
    return md


def _value_to_md(value, level: int, label: str) -> str:
    if isinstance(value, dict):
        return _record_to_md(value, level)
    if isinstance(value, list):
        return _list_to_md(value, level, label)
    return _scalar_value_to_md(value)


def _dict_to_md(filename: str, data: list[dict], is_truncated: bool) -> str:
    md = f"# {filename}\n\n"
    if is_truncated:
        md += f"Showing the first {len(data)} records of the file.\n\n"
    for index, record in enumerate(data, start=1):
        md += "---\n\n"
        md += _heading(2, f"Record {index}")
        md += _value_to_md(record, 3, f"Record {index}")
    return md


def _read_jsonl_file(jsonl_file_path: Path, max_records: int | None) -> list[dict]:
    records = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            if max_records is not None and len(records) >= max_records:
                break
            records.append(json.loads(line))
    return records


def _write_md_file(md_file_path: Path, md: str) -> None:
    with open(md_file_path, "w") as f:
        f.write(md)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="The JSONL file to convert to MD.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first LIMIT lines of the JSONL file. Defaults to every line.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError(f"The limit {args.limit} must be at least 1.")

    jsonl_file_path = Path(args.file)
    if not jsonl_file_path.exists():
        raise FileNotFoundError(f"The JSONL file {jsonl_file_path} does not exist.")
    if not jsonl_file_path.is_file():
        raise IsADirectoryError(f"The JSONL file {jsonl_file_path} is a directory.")
    if not jsonl_file_path.suffix == ".jsonl":
        raise ValueError(f"The JSONL file {jsonl_file_path} is not a JSONL file.")

    data = _read_jsonl_file(jsonl_file_path, args.limit)

    is_truncated = args.limit is not None and len(data) == args.limit
    md = _dict_to_md(jsonl_file_path.name, data, is_truncated)

    md_file_path = jsonl_file_path.with_suffix(".md")
    if md_file_path.exists():
        print(f"{md_file_path} already exists. Overwriting...")

    _write_md_file(md_file_path, md)
    print(f"{md_file_path} has been created successfully.")


if __name__ == "__main__":
    main()
