import os
import json

def save_to_json(data: list[dict], filename: str):
    os.makedirs("data", exist_ok=True)
    with open(f"data/{filename}.json", "w") as f:
        json.dump(data, f, indent=2)