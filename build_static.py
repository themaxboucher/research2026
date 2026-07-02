"""Build a single self-contained HTML page that shows, for each comment
generation target, the human's diff hunk side-by-side with every model's
generated comment. The result is fully static — no server, suitable for
hosting on Vercel or any static host."""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from runs import require_latest_run_directory

# The prompt wraps the diff in an XML-ish tag. Convert it to a fenced markdown
# code block so the dashboard can syntax-highlight it.
_CHANGE_RE = re.compile(r"<change>\s*\n(.*?)\n\s*</change>", re.DOTALL)


def _prompt_to_markdown(prompt: str) -> str:
    return _CHANGE_RE.sub(lambda m: f"```diff\n{m.group(1)}\n```", prompt)

TEMPLATE_PATH = Path(__file__).parent / "static_template.html"
DEFAULT_OUTPUT = Path(__file__).parent / "dist" / "index.html"
GENERATED_FILENAME = "files_generated.jsonl"


def _short_model_name(model: str) -> str:
    return model.split("/")[-1] if model else "(unknown)"


def _trim_target(record: dict, generation: dict) -> dict:
    """Pull just the fields needed for the comparison view."""
    results = []
    prompt_text = ""
    for result in generation.get("results") or []:
        if not prompt_text and result.get("prompt"):
            prompt_text = result["prompt"]
        results.append(
            {
                "model": _short_model_name(result.get("model") or ""),
                "comment": result.get("comment_text") or "",
                "error": result.get("error"),
            }
        )

    return {
        "repo": record.get("repo_name") or "",
        "commit": (record.get("commit_hash") or "")[:7],
        "path": record.get("new_path") or record.get("filename") or "",
        "type": generation.get("type") or "",
        "status": generation.get("status") or "",
        "anchor": generation.get("anchor") or "",
        "start_line": generation.get("start_line"),
        "end_line": generation.get("end_line"),
        "prompt": _prompt_to_markdown(prompt_text),
        "human": generation.get("comment") or "",
        "results": results,
    }


def _collect_targets(generated_path: Path) -> tuple[list[dict], set[str]]:
    targets: list[dict] = []
    models: set[str] = set()
    with generated_path.open("r", encoding="utf-8") as source_file:
        for raw_line in source_file:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                logging.warning("Skipping unparseable line in %s", generated_path)
                continue
            for generation in record.get("comment_generations") or []:
                target = _trim_target(record, generation)
                for result in target["results"]:
                    if result["model"]:
                        models.add(result["model"])
                targets.append(target)
    return targets, models


def _build_payload(run_dir: Path) -> dict:
    generated_path = run_dir / GENERATED_FILENAME
    if not generated_path.exists():
        raise FileNotFoundError(f"No {GENERATED_FILENAME} in {run_dir}")
    targets, models = _collect_targets(generated_path)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(run_dir.name),
        "models": sorted(models),
        "targets": targets,
    }


def _render(template: str, payload: dict) -> str:
    # Embed as a JSON string parsed at runtime — much faster than parsing a
    # giant JS object literal and avoids </script> escaping concerns.
    data_json = json.dumps(payload, ensure_ascii=False)
    # Defensive: the only sequence that breaks an embedded JSON-in-script is
    # the literal "</" closing the script tag; escape the slash.
    data_json = data_json.replace("</", "<\\/")
    return template.replace("__DATA_JSON__", data_json)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Build a static HTML comparison view of comment generations."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Run directory (defaults to latest run).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir if args.run_dir is not None else require_latest_run_directory()
    logging.info("Building static comparison view from %s", run_dir)
    payload = _build_payload(run_dir)
    logging.info(
        "Collected %d targets across %d models",
        len(payload["targets"]),
        len(payload["models"]),
    )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = _render(template, payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    logging.info("Wrote %s (%.1f KB)", args.output, size_kb)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(sys.argv[1:])
