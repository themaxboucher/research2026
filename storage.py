import json
from pathlib import Path


def save_to_json(data: list[dict], directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{filename}.json"
    with open(output_path, "w") as output_file:
        json.dump(data, output_file, indent=2)


def load_from_json(directory: Path, filename: str) -> list[dict]:
    input_path = directory / f"{filename}.json"
    with open(input_path, "r") as input_file:
        return json.load(input_file)
