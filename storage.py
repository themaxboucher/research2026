import json
import os
import shutil
from collections.abc import Callable
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


def _start_of_last_line(jsonl_file, end: int) -> int:
    search_end = end
    jsonl_file.seek(end - 1)
    if jsonl_file.read(1) == b"\n":
        search_end = end - 1

    chunk_size = 8192
    position = search_end
    while position > 0:
        read_start = max(0, position - chunk_size)
        jsonl_file.seek(read_start)
        chunk = jsonl_file.read(position - read_start)
        newline_index = chunk.rfind(b"\n")
        if newline_index != -1:
            return read_start + newline_index + 1
        position = read_start
    return 0


def _is_valid_json(line: bytes) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True


def truncate_broken_tail(directory: Path, filename: str) -> int:
    jsonl_path = directory / f"{filename}.jsonl"
    if not jsonl_path.exists():
        return 0

    with jsonl_path.open("r+b") as jsonl_file:
        size = jsonl_file.seek(0, os.SEEK_END)
        end = size
        while end > 0:
            line_start = _start_of_last_line(jsonl_file, end)
            jsonl_file.seek(line_start)
            line = jsonl_file.read(end - line_start)
            if line.endswith(b"\n") and _is_valid_json(line):
                break
            end = line_start
        jsonl_file.truncate(end)
    return size - end


def drop_trailing_records(
    directory: Path, filename: str, should_drop: Callable[[dict], bool]
) -> int:
    jsonl_path = directory / f"{filename}.jsonl"
    if not jsonl_path.exists():
        return 0

    removed_records = 0
    with jsonl_path.open("r+b") as jsonl_file:
        end = jsonl_file.seek(0, os.SEEK_END)
        while end > 0:
            line_start = _start_of_last_line(jsonl_file, end)
            jsonl_file.seek(line_start)
            record = json.loads(jsonl_file.read(end - line_start))
            if not should_drop(record):
                break
            end = line_start
            removed_records += 1
        jsonl_file.truncate(end)
    return removed_records


def merge_jsonl_shards(
    directory: Path, filename: str, *, delete_shards: bool = False
) -> int:
    shard_paths = sorted(directory.glob(f"{filename}.*.jsonl"))
    output_path = directory / f"{filename}.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8") as shard_file:
                shutil.copyfileobj(shard_file, output_file)
    if delete_shards:
        for shard_path in shard_paths:
            shard_path.unlink()
    return len(shard_paths)


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
