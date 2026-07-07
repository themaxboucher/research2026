"""A *run* holds a shared source dataset plus one or more *generations*. Each
generation is a subdirectory under `runs/<ts>/generations/<slug>/` holding its
own model output (`files_generated.jsonl`), manual review notes
(`review_notes.json`), a manifest describing how it was produced, and the
dashboard's index caches. Generations in a run share the run's
`repo_files_sample.jsonl` and differ only in prompt/model/config."""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GENERATIONS_DIRNAME = "generations"
MANIFEST_FILENAME = "generation.json"
GENERATED_FILENAME = "files_generated"
NOTES_FILENAME = "review_notes.json"


def _slugify(label: str) -> str:
    """Filesystem-safe directory name for a user-supplied generation label."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-._")
    return slug or "generation"


def default_label() -> str:
    """Timestamp label used when the caller doesn't name the generation."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def generations_root(run_dir: Path) -> Path:
    return run_dir / GENERATIONS_DIRNAME


def generation_dir(run_dir: Path, label: str) -> Path:
    return generations_root(run_dir) / _slugify(label)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip() or None


def write_manifest(
    gen_dir: Path,
    label: str,
    model_profile: str | None,
    model_names: list[str],
    config: dict,
) -> dict:
    """Record how a generation was produced so it can be told apart from others
    in the run (models, code version, limits) long after the fact."""
    manifest = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_profile": model_profile,
        "model_names": list(model_names),
        "git_commit": _git_commit(),
        "config": config,
    }
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def read_manifest(gen_dir: Path) -> dict:
    manifest_path = gen_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def list_generations(run_dir: Path) -> list[dict]:
    """Every generation in the run that has produced output, newest first. Each
    entry carries the manifest fields the dashboard selector needs plus the
    generation's directory path."""
    root = generations_root(run_dir)
    if not root.exists():
        return []

    generations: list[dict] = []
    for gen_dir in root.iterdir():
        if not gen_dir.is_dir():
            continue
        if not (gen_dir / f"{GENERATED_FILENAME}.jsonl").exists():
            continue
        manifest = read_manifest(gen_dir)
        generations.append(
            {
                "id": gen_dir.name,
                "label": manifest.get("label") or gen_dir.name,
                "created_at": manifest.get("created_at"),
                "model_profile": manifest.get("model_profile"),
                "model_names": manifest.get("model_names") or [],
                "config": manifest.get("config") or {},
                "dir": gen_dir,
            }
        )

    # Newest first. Missing timestamps sort last so named-but-manifestless
    # generations don't crowd out dated ones.
    generations.sort(key=lambda gen: gen.get("created_at") or "", reverse=True)
    return generations
