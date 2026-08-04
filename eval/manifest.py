import json
from pathlib import Path

EVAL_MANIFEST_FILENAME = "eval"


def write_eval_manifest(run_dir: Path, *, num_tasks: int) -> dict:
    manifest = {"num_tasks": num_tasks}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{EVAL_MANIFEST_FILENAME}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def read_eval_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / f"{EVAL_MANIFEST_FILENAME}.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}
