import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "files.json"


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        files = json.load(f)

    total = sum(len(entry.get("comments", [])) for entry in files)
    print(total)


if __name__ == "__main__":
    main()
