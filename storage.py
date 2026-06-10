import json
from pathlib import Path


def _write_jsonl(
    data: list[dict], directory: Path, filename: str, *, mode: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{filename}.jsonl"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for record in data:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_to_jsonl(data: list[dict], directory: Path, filename: str) -> None:
    _write_jsonl(data, directory, filename, mode="w")


def append_to_jsonl(data: list[dict], directory: Path, filename: str) -> None:
    _write_jsonl(data, directory, filename, mode="a")


def iter_from_jsonl(directory: Path, filename: str):
    jsonl_path = directory / f"{filename}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No dataset file found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                yield json.loads(stripped_line)


def load_from_jsonl(directory: Path, filename: str) -> list[dict]:
    jsonl_path = directory / f"{filename}.jsonl"
    json_path = directory / f"{filename}.json"

    if jsonl_path.exists():
        records: list[dict] = []
        with jsonl_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                stripped_line = line.strip()
                if stripped_line:
                    records.append(json.loads(stripped_line))
        return records

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    raise FileNotFoundError(f"No dataset file found: {jsonl_path} or {json_path}")
