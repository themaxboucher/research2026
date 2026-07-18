import argparse
import copy
import difflib
import json
import logging
import os
import re
import shutil
import socket
import sys
import textwrap
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import generate.generate as generate
from generate.comments import apply_generated_comment, is_machine_directive_comment
from collect.dataset import latest_dataset_directory

DATASET_FILENAME = "dataset_sample"
REGENERATED_METRICS_FILENAME = "metrics_regenerate.json"
NOTES_FILENAME = "review_notes.json"
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"
DEFAULT_PAGE_LIMIT = 200
MAX_PAGE_LIMIT = 1000

# The intent labels a human comment can carry (why the code is as it is, what
# the code does, how it works, or something else). Stored on the comment in the
# source dataset.
INTENT_VALUES = ("why", "what", "how", "other")

# Bump when the cached index schema changes so stale caches are rebuilt.
INDEX_VERSION = 6

SIDEBAR_FIELDS = (
    "repo_name",
    "commit_hash",
    "filename",
    "new_path",
    "change_type",
    "nloc",
    "added_lines",
    "deleted_lines",
)

# Comments surfaced inline under each file in the sidebar.
SIDEBAR_COMMENT_TYPES = {"inline", "block"}
SIDEBAR_COMMENT_STATUSES = {"added", "modified"}
SIDEBAR_COMMENT_FIELDS = ("type", "status", "start_line", "end_line", "comment", "intent")

META_FIELDS = (
    "count",
    "repo_count",
    "comment_count",
    "sidebar_comment_count",
    "change_types",
)


def _comment_generation_count(gen_record: dict, comment: dict) -> int | None:
    """Number of model results the generation record holds for `comment`, or
    None when the generation didn't target it. Matched by the comment's
    identity fields (type and line span)."""
    for generation in gen_record.get("comment_generations") or []:
        if (
            generation.get("type") == comment.get("type")
            and generation.get("start_line") == comment.get("start_line")
            and generation.get("end_line") == comment.get("end_line")
        ):
            return len(generation.get("results") or [])
    return None


def _sidebar_comments(record: dict) -> list[dict]:
    """The added/modified inline/block comments shown under a file in the
    sidebar."""
    comments = []
    for comment in record.get("comments") or []:
        if comment.get("type") not in SIDEBAR_COMMENT_TYPES:
            continue
        if comment.get("status") not in SIDEBAR_COMMENT_STATUSES:
            continue
        comments.append({field: comment.get(field) for field in SIDEBAR_COMMENT_FIELDS})
    return comments


def _index_paths(run_dir: Path, dataset: str) -> tuple[Path, Path]:
    index_path = run_dir / f"{dataset}.index.jsonl"
    meta_path = run_dir / f"{dataset}.index.meta.json"
    return index_path, meta_path


def _build_index(source_path: Path) -> tuple[list[dict], dict]:
    """Stream the source file once, recording each record's byte offset plus
    the small sidebar fields. Stops at the first unparseable line, assuming a
    truncated tail from an interrupted run."""
    sidebar: list[dict] = []
    repos: set[str] = set()
    change_types: set[str] = set()
    comment_total = 0
    sidebar_comment_total = 0
    offset = 0

    with source_path.open("rb") as source_file:
        for raw_line in source_file:
            line_start = offset
            offset += len(raw_line)
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                logging.warning(
                    "Stopping index build at byte offset %d: line did not parse "
                    "(assuming truncated tail).",
                    line_start,
                )
                break

            entry = {"index": len(sidebar), "offset": line_start}
            for field in SIDEBAR_FIELDS:
                entry[field] = record.get(field)
            entry["comment_count"] = len(record.get("comments") or [])
            entry["sidebar_comments"] = _sidebar_comments(record)
            sidebar_comment_total += len(entry["sidebar_comments"])
            sidebar.append(entry)

            if entry["repo_name"]:
                repos.add(entry["repo_name"])
            if entry["change_type"]:
                change_types.add(entry["change_type"])
            comment_total += entry["comment_count"]

    meta = {
        "count": len(sidebar),
        "repo_count": len(repos),
        "comment_count": comment_total,
        "sidebar_comment_count": sidebar_comment_total,
        "change_types": sorted(change_types),
    }
    return sidebar, meta


def _write_cache(
    index_path: Path, meta_path: Path, sidebar: list[dict], meta: dict, stat
) -> None:
    with index_path.open("w", encoding="utf-8") as index_file:
        for entry in sidebar:
            index_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    meta_out = {field: meta[field] for field in META_FIELDS}
    meta_out["source"] = {"size": stat.st_size, "mtime": stat.st_mtime}
    meta_out["version"] = INDEX_VERSION
    meta_path.write_text(json.dumps(meta_out), encoding="utf-8")


def _load_valid_cache(source_path: Path, dataset: str) -> tuple[list[dict], dict] | None:
    index_path, meta_path = _index_paths(source_path.parent, dataset)
    if not index_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("version") != INDEX_VERSION:
        return None
    stat = source_path.stat()
    source = meta.get("source", {})
    if source.get("size") != stat.st_size or source.get("mtime") != stat.st_mtime:
        return None

    sidebar: list[dict] = []
    with index_path.open("r", encoding="utf-8") as index_file:
        for line in index_file:
            stripped = line.strip()
            if stripped:
                sidebar.append(json.loads(stripped))
    return sidebar, {field: meta[field] for field in META_FIELDS}


def _load_index(run_dir: Path, dataset: str) -> tuple[Path, list[dict], dict]:
    source_path = run_dir / f"{dataset}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"No dataset file found: {source_path}")

    cached = _load_valid_cache(source_path, dataset)
    if cached is not None:
        sidebar, meta = cached
        logging.info("Loaded cached index for %s (%d records)", source_path, meta["count"])
        return source_path, sidebar, meta

    stat = source_path.stat()
    logging.info("Building index for %s (%d bytes)...", source_path, stat.st_size)
    sidebar, meta = _build_index(source_path)
    index_path, meta_path = _index_paths(run_dir, dataset)
    _write_cache(index_path, meta_path, sidebar, meta, stat)
    logging.info("Indexed %d records into %s", meta["count"], index_path)
    return source_path, sidebar, meta


def _record_key(entry: dict) -> tuple:
    return (
        entry.get("repo_name"),
        entry.get("commit_hash"),
        entry.get("new_path") or entry.get("filename"),
    )


def _match_generated(sidebar: list[dict], generated_records: dict[tuple, dict]) -> int:
    """Attach generation info to sidebar entries, joining each slim generation
    record to its source file by the (repo, commit, path) key it carries."""
    matched = 0
    for entry in sidebar:
        gen_record = generated_records.get(_record_key(entry))
        if gen_record is None:
            continue
        entry["generation_count"] = len(gen_record.get("comment_generations") or [])
        for comment in entry.get("sidebar_comments") or []:
            count = _comment_generation_count(gen_record, comment)
            if count is not None:
                comment["generation_model_count"] = count
        matched += 1
    return matched


def _attach_generation_diffs(record: dict) -> None:
    """For each model result under each comment generation, rebuild the model's
    patched source (generation records store only the comment text) and attach
    a unified diff of the human source vs it. Since the patched source differs
    only at the target comment, this is a single small hunk that the dashboard
    renders as the model's diff hunk."""
    source_code = record.get("source_code") or ""
    filepath = record.get("new_path") or record.get("filename") or "file.py"
    for generation in record.get("comment_generations") or []:
        for result in generation.get("results") or []:
            comment_text = result.get("comment_text")
            if not comment_text:
                result["diff"] = ""
                continue
            try:
                new_source_code = apply_generated_comment(
                    source_code, generation, comment_text
                )
            except Exception:
                logging.warning(
                    "Could not rebuild patched source for %s", filepath, exc_info=True
                )
                result["diff"] = ""
                continue
            diff_lines = difflib.unified_diff(
                source_code.splitlines(keepends=True),
                new_source_code.splitlines(keepends=True),
                fromfile=f"a/{filepath}",
                tofile=f"b/{filepath}",
            )
            result["diff"] = "".join(diff_lines)


def _read_record(source_path: Path, offset: int) -> dict:
    with source_path.open("rb") as source_file:
        source_file.seek(offset)
        line = source_file.readline()
    return json.loads(line)


def _display_order(sidebar: list[dict]) -> list[dict]:
    generated = [entry for entry in sidebar if "generation_count" in entry]
    ungenerated = [entry for entry in sidebar if "generation_count" not in entry]
    return generated + ungenerated


def _query_records(ordered_sidebar: list[dict], params: dict[str, list[str]]) -> dict:
    query = (params.get("q", [""])[0] or "").strip().lower()
    types_param = params.get("types", [None])[0]
    active_types = set(types_param.split(",")) if types_param else None
    generated_only = params.get("gen", ["0"])[0] == "1"

    try:
        offset = max(0, int(params.get("offset", ["0"])[0]))
    except ValueError:
        offset = 0
    try:
        limit = int(params.get("limit", [str(DEFAULT_PAGE_LIMIT)])[0])
    except ValueError:
        limit = DEFAULT_PAGE_LIMIT
    limit = max(1, min(limit, MAX_PAGE_LIMIT))

    matched: list[dict] = []
    for entry in ordered_sidebar:
        if generated_only and "generation_count" not in entry:
            continue
        if active_types is not None and entry.get("change_type") not in active_types:
            continue
        if query:
            haystack = " ".join(
                value
                for value in (
                    entry.get("filename"),
                    entry.get("new_path"),
                    entry.get("repo_name"),
                )
                if value
            ).lower()
            if query not in haystack:
                continue
        matched.append(entry)

    return {"rows": matched[offset : offset + limit], "total": len(matched)}


# --- Comment identity & intent -------------------------------------------
# A comment's stable identity across the source, the generations, and the
# review/intent stores: repo, short commit, path, type, and line span. The same
# string is used as the key everywhere so intent labels attach consistently.


def _comment_key(record: dict, comment: dict) -> str:
    repo = record.get("repo_name") or ""
    commit = (record.get("commit_hash") or "")[:7]
    path = record.get("new_path") or record.get("filename") or ""
    return "|".join(
        str(part)
        for part in (
            repo,
            commit,
            path,
            comment.get("type") or "",
            comment.get("start_line"),
            comment.get("end_line"),
        )
    )


def _is_intent_target(comment: dict) -> bool:
    """The comments eligible for intent labeling: the same set the generation
    pipeline targets (added inline/block comments that aren't machine
    directives)."""
    return (
        comment.get("type") in SIDEBAR_COMMENT_TYPES
        and comment.get("status") == "added"
        and comment.get("comment") is not None
        and not is_machine_directive_comment(comment["comment"])
    )


def _stamp_sidebar_intents(sidebar: list[dict], intent_map: dict[str, str]) -> None:
    """Overlay the source's intent labels onto sidebar comments. Needed because a
    generation's copied comments predate labeling and so lack intent."""
    if not intent_map:
        return
    for entry in sidebar:
        for comment in entry.get("sidebar_comments") or []:
            comment["intent"] = intent_map.get(_comment_key(entry, comment))


def _stamp_record_intents(record: dict, intent_map: dict[str, str]) -> None:
    """Overlay intent labels onto a full record's comments, so the comment-detail
    panel shows them even for a generation's (pre-labeling) copied comments."""
    if not intent_map:
        return
    for comment in record.get("comments") or []:
        key = _comment_key(record, comment)
        if key in intent_map:
            comment["intent"] = intent_map[key]


# --- Generations analysis view -------------------------------------------
# The "Generations" view is a flat comparison of every comment-generation
# target: the prompt sent to the models, the human's comment, and each model's
# generated comment side by side. Built once from the generation's in-memory
# location_generated records and cached; served as a single JSON payload.

# The prompt wraps the diff in an XML-ish tag. Convert it to a fenced markdown
# code block so the client can syntax-highlight it.
_CHANGE_RE = re.compile(r"<change>\s*\n(.*?)\n\s*</change>", re.DOTALL)


def _prompt_to_markdown(prompt: str) -> str:
    return _CHANGE_RE.sub(lambda m: f"```diff\n{m.group(1)}\n```", prompt)


def _short_model_name(model: str) -> str:
    return model.split("/")[-1] if model else "(unknown)"


def _trim_target(record: dict, generation: dict, intent_map: dict[str, str]) -> dict:
    """Pull just the fields needed for the comparison view."""
    results = []
    for result in generation.get("results") or []:
        trimmed = {
            "model": _short_model_name(result.get("model") or ""),
            "comment": result.get("comment_text") or "",
            "error": result.get("error"),
        }
        # eval.py writes scores onto scored results (null when the prediction
        # was unusable); leave the key absent when eval.py hasn't run so the
        # client can tell the two apart.
        if "scores" in result:
            trimmed["scores"] = result["scores"]
        results.append(trimmed)

    prompt_text = generation.get("prompt") or ""

    repo = record.get("repo_name") or ""
    commit_full = record.get("commit_hash") or ""
    commit = commit_full[:7]
    path = record.get("new_path") or record.get("filename") or ""
    # Stable identity for attaching review notes and intent, robust to reordering.
    key = _comment_key(record, generation)

    return {
        "key": key,
        "repo": repo,
        "commit": commit,
        "commit_full": commit_full,
        "path": path,
        "type": generation.get("type") or "",
        "status": generation.get("status") or "",
        "anchor": generation.get("anchor") or "",
        "start_line": generation.get("start_line"),
        "end_line": generation.get("end_line"),
        "prompt": _prompt_to_markdown(prompt_text),
        "human": generation.get("comment") or "",
        "intent": intent_map.get(key) if intent_map else None,
        "results": results,
    }


def _collect_targets(
    generated_records: list[dict], intent_map: dict[str, str]
) -> tuple[list[dict], set[str]]:
    targets: list[dict] = []
    models: set[str] = set()
    for record in generated_records:
        for generation in record.get("comment_generations") or []:
            target = _trim_target(record, generation, intent_map)
            for result in target["results"]:
                if result["model"]:
                    models.add(result["model"])
            targets.append(target)
    return targets, models


def _build_generations_payload(
    generated_records: list[dict], source_name: str, intent_map: dict[str, str]
) -> dict:
    targets, models = _collect_targets(generated_records, intent_map)
    logging.info(
        "Built generations view: %d targets across %d models",
        len(targets),
        len(models),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_name,
        "models": sorted(models),
        "targets": targets,
    }


# --- Regenerations view ----------------------------------------------------
# The "Regenerate" sub-view of the Generations view: one card per scope that
# was rewritten by the models with comments added. Each model result carries a
# diff of the stripped input vs the regenerated code (only comment additions
# for valid regenerations) plus the per-target extractions eval.py scored.
# Built once per generation from regenerate_generated.jsonl and cached.


def _regeneration_diff(input_code: str, regenerated_code: str, filepath: str) -> str:
    """Unified diff of the stripped input scope vs the model's regeneration.
    Both sides are dedented first so a model that uniformly dedented an
    indented scope (accepted by the fidelity check) doesn't diff as a full
    rewrite."""
    diff_lines = difflib.unified_diff(
        textwrap.dedent(input_code).splitlines(keepends=True),
        textwrap.dedent(regenerated_code).splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff_lines)


def _trim_regeneration_result(result: dict, input_code: str, filepath: str) -> dict:
    error = result.get("error")
    trimmed = {
        "model": _short_model_name(result.get("model") or ""),
        "error": error,
    }
    if error:
        # The raw response is what you need to see to understand why the
        # fidelity check rejected the regeneration.
        trimmed["raw_response"] = result.get("raw_response") or ""
        return trimmed

    trimmed["diff"] = _regeneration_diff(
        input_code, result.get("regenerated_code") or "", filepath
    )
    extractions = []
    for extraction in result.get("extractions") or []:
        entry = {
            "comment_text": extraction.get("comment_text"),
            "placement_hit": extraction.get("placement_hit"),
            "form_matches": extraction.get("form_matches"),
            "error": extraction.get("error"),
        }
        if "scores" in extraction:
            entry["scores"] = extraction["scores"]
        extractions.append(entry)
    trimmed["extractions"] = extractions
    trimmed["hit_count"] = sum(
        1 for extraction in extractions if extraction["placement_hit"]
    )
    return trimmed


def _trim_scope(record: dict, intent_map: dict[str, str]) -> dict:
    repo = record.get("repo_name") or ""
    commit_full = record.get("commit_hash") or ""
    path = record.get("new_path") or ""
    input_code = record.get("input_code") or ""

    targets = []
    for target in record.get("targets") or []:
        key = _comment_key(
            {"repo_name": repo, "commit_hash": commit_full, "new_path": path}, target
        )
        targets.append(
            {
                "type": target.get("type") or "",
                "intent": intent_map.get(key) or target.get("intent"),
                "start_line": target.get("start_line"),
                "end_line": target.get("end_line"),
                "anchor": target.get("anchor") or "",
                "comment": target.get("comment") or "",
            }
        )

    return {
        "key": "|".join(
            str(part)
            for part in (
                repo,
                commit_full[:7],
                path,
                record.get("scope_start_line"),
                record.get("scope_end_line"),
            )
        ),
        "repo": repo,
        "commit": commit_full[:7],
        "commit_full": commit_full,
        "path": path,
        "scope_start_line": record.get("scope_start_line"),
        "scope_end_line": record.get("scope_end_line"),
        "prompt": record.get("prompt") or "",
        "targets": targets,
        "results": [
            _trim_regeneration_result(result, input_code, path)
            for result in record.get("results") or []
        ],
    }


def _load_regeneration_metrics(gen_dir: Path) -> dict | None:
    metrics_path = gen_dir / REGENERATED_METRICS_FILENAME
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {_short_model_name(model): values for model, values in metrics.items()}


def _empty_regenerations_payload(source_name: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_name,
        "models": [],
        "scopes": [],
        "metrics": None,
    }


def _build_regenerations_payload(
    gen_dir: Path, source_name: str, intent_map: dict[str, str]
) -> dict:
    regenerated_path = gen_dir / f"{generate.REGENERATE_FILENAME}.jsonl"
    if not regenerated_path.exists():
        return _empty_regenerations_payload(source_name)

    scopes: list[dict] = []
    models: set[str] = set()
    for record in _iter_source_records(regenerated_path):
        scope = _trim_scope(record, intent_map)
        for result in scope["results"]:
            if result["model"]:
                models.add(result["model"])
        scopes.append(scope)

    logging.info(
        "Built regenerations view: %d scopes across %d models", len(scopes), len(models)
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_name,
        "models": sorted(models),
        "scopes": scopes,
        "metrics": _load_regeneration_metrics(gen_dir),
    }


# --- Intent labeling view -------------------------------------------------
# Labeling each human comment's intent (why the code is as it is / what it does /
# how it works). Targets come from the source dataset (shared across
# generations); labels are written back onto the comments in
# dataset_sample.jsonl.


def _iter_source_records(source_path: Path):
    with source_path.open("r", encoding="utf-8") as source_file:
        for raw_line in source_file:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except ValueError:
                logging.warning("Skipping unparseable line in %s", source_path)
                continue


def _build_intent_map(source_path: Path) -> dict[str, str]:
    """{comment key -> intent} for every source comment that carries a label."""
    intent_map: dict[str, str] = {}
    for record in _iter_source_records(source_path):
        for comment in record.get("comments") or []:
            if comment.get("intent") in INTENT_VALUES:
                intent_map[_comment_key(record, comment)] = comment["intent"]
    return intent_map


def _build_intents_payload(source_path: Path) -> dict:
    """The intent-labeling view: every target comment with its full source file
    and the comment's absolute line span, plus the comment's current label. The
    client renders a small window around the comment and expands to the full file
    on demand."""
    targets: list[dict] = []
    for record in _iter_source_records(source_path):
        source_code = record.get("source_code") or ""
        for comment in record.get("comments") or []:
            if not _is_intent_target(comment):
                continue
            intent = comment.get("intent")
            targets.append(
                {
                    "key": _comment_key(record, comment),
                    "repo": record.get("repo_name") or "",
                    "commit": (record.get("commit_hash") or "")[:7],
                    "commit_full": record.get("commit_hash") or "",
                    "path": record.get("new_path") or record.get("filename") or "",
                    "type": comment.get("type") or "",
                    "status": comment.get("status") or "",
                    "anchor": comment.get("anchor") or "",
                    "start_line": comment.get("start_line"),
                    "end_line": comment.get("end_line"),
                    "comment": comment.get("comment") or "",
                    "source_code": source_code,
                    "comment_start": comment.get("start_line"),
                    "comment_end": comment.get("end_line"),
                    "intent": intent if intent in INTENT_VALUES else None,
                }
            )
    logging.info("Built intent view: %d target comments", len(targets))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_path.parent.name,
        "targets": targets,
    }


def _write_intents_to_source(source_path: Path, intents: dict[str, str]) -> int:
    """Write intent labels back onto the matching source comments. Backs the file
    up once, then rewrites it atomically. `intents` is the client's full view of
    every comment it has an opinion on (a label, or null to clear). Returns the
    number of comments whose label changed."""
    backup_path = source_path.parent / (source_path.name + ".bak")
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)
        logging.info("Backed up source dataset to %s", backup_path)

    changed = 0
    tmp_path = source_path.parent / (source_path.name + ".tmp")
    with (
        source_path.open("r", encoding="utf-8") as source_file,
        tmp_path.open("w", encoding="utf-8") as tmp_file,
    ):
        for raw_line in source_file:
            stripped = raw_line.strip()
            if not stripped:
                tmp_file.write(raw_line)
                continue
            record = json.loads(stripped)
            for comment in record.get("comments") or []:
                key = _comment_key(record, comment)
                if key not in intents:
                    continue
                new_intent = intents[key] or None
                if new_intent is not None and new_intent not in INTENT_VALUES:
                    continue  # ignore unknown label values
                if comment.get("intent") != new_intent:
                    changed += 1
                if new_intent is None:
                    comment.pop("intent", None)
                else:
                    comment["intent"] = new_intent
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, source_path)
    logging.info("Wrote %d intent change(s) to %s", changed, source_path)
    return changed


def _build_generation_state(
    gen_dir: Path, base_sidebar: list[dict], intent_map: dict[str, str]
) -> dict:
    """Everything the handler needs to serve one generation: its slim records
    held in memory keyed for joining against the source, a private copy of the
    source sidebar annotated with this generation's counts, the display-ordered
    view of it, and where its notes live. Built lazily and cached per generation
    so switching is cheap and generations never contaminate each other's sidebar
    annotations. A generation produced with only the regenerate approach has no
    location_generated.jsonl; it serves the bare source sidebar and an empty
    per-target comparison."""
    sidebar = copy.deepcopy(base_sidebar)
    generated_records = None
    location_path = gen_dir / f"{generate.LOCATION_FILENAME}.jsonl"
    if location_path.exists():
        generated_records = {
            _record_key(record): record for record in _iter_source_records(location_path)
        }
        matched = _match_generated(sidebar, generated_records)
        if matched < len(generated_records):
            logging.warning(
                "%d generated record(s) have no dataset match in generation %s",
                len(generated_records) - matched,
                gen_dir.name,
            )
        logging.info(
            "Matched %d of %d generated records for generation %s",
            matched,
            len(generated_records),
            gen_dir.name,
        )
    # Sidebar comments come from the source index, which may predate the latest
    # labeling pass, so re-stamp intent to keep the chips current.
    _stamp_sidebar_intents(sidebar, intent_map)
    return {
        "gen_dir": gen_dir,
        "generated_records": generated_records,
        "sidebar": sidebar,
        "ordered_sidebar": _display_order(sidebar),
        "notes_path": gen_dir / NOTES_FILENAME,
        "payload": None,  # generations comparison payload, built on first request
        "regen_payload": None,  # regenerations payload, built on first request
    }


def _make_handler(
    html_bytes: bytes,
    source_path: Path,
    base_sidebar: list[dict],
    base_ordered: list[dict],
    meta: dict,
    run_dir: Path,
    generations_by_id: dict[str, dict],
    default_gen_id: str | None,
    states: dict[str, dict],
    intent_map: dict[str, str],
    intents_cache: dict,
):
    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logging.info("%s - %s", self.address_string(), format % args)

        def _send_json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def _resolve_gen_id(self, params: dict[str, list[str]]) -> str | None:
            """The generation the request targets: the `generation` query param
            when it names a real generation, else the newest (default). (`gen`
            is already taken by the records 'generated only' filter.)"""
            requested = params.get("generation", [None])[0]
            if requested and requested in generations_by_id:
                return requested
            return default_gen_id

        def _generation_state(self, gen_id: str | None) -> dict | None:
            if gen_id is None or gen_id not in generations_by_id:
                return None
            if gen_id not in states:
                states[gen_id] = _build_generation_state(
                    generations_by_id[gen_id]["dir"], base_sidebar, intent_map
                )
            return states[gen_id]

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            params = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self._send_html()
                return

            if path == "/meta":
                self._send_json(meta)
                return

            if path == "/records":
                state = self._generation_state(self._resolve_gen_id(params))
                ordered = state["ordered_sidebar"] if state else base_ordered
                self._send_json(_query_records(ordered, params))
                return

            if path == "/generations":
                state = self._generation_state(self._resolve_gen_id(params))
                if state is None or state["generated_records"] is None:
                    self._send_json(
                        {
                            "generated_at": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"
                            ),
                            "source": run_dir.name,
                            "models": [],
                            "targets": [],
                        }
                    )
                    return
                if state["payload"] is None:
                    state["payload"] = _build_generations_payload(
                        list(state["generated_records"].values()),
                        run_dir.name,
                        intent_map,
                    )
                self._send_json(state["payload"])
                return

            if path == "/regenerations":
                state = self._generation_state(self._resolve_gen_id(params))
                if state is None:
                    self._send_json(_empty_regenerations_payload(run_dir.name))
                    return
                if state["regen_payload"] is None:
                    state["regen_payload"] = _build_regenerations_payload(
                        state["gen_dir"], run_dir.name, intent_map
                    )
                self._send_json(state["regen_payload"])
                return

            if path == "/intents":
                if intents_cache.get("payload") is None:
                    intents_cache["payload"] = _build_intents_payload(source_path)
                self._send_json(intents_cache["payload"])
                return

            if path == "/notes":
                state = self._generation_state(self._resolve_gen_id(params))
                notes_path = state["notes_path"] if state else None
                if notes_path is not None and notes_path.exists():
                    self._send_json(json.loads(notes_path.read_text(encoding="utf-8")))
                else:
                    self._send_json(
                        {"run": run_dir.name, "failure_modes": [], "entries": {}}
                    )
                return

            if path.startswith("/records/"):
                try:
                    index = int(path[len("/records/") :])
                except ValueError:
                    self._send_json({"error": "invalid index"}, HTTPStatus.BAD_REQUEST)
                    return
                state = self._generation_state(self._resolve_gen_id(params))
                sidebar = state["sidebar"] if state else base_sidebar
                if not (0 <= index < len(sidebar)):
                    self._send_json({"error": "out of range"}, HTTPStatus.NOT_FOUND)
                    return
                entry = sidebar[index]
                record = _read_record(source_path, entry["offset"])
                if state is not None and state["generated_records"]:
                    gen_record = state["generated_records"].get(_record_key(entry))
                    if gen_record is not None:
                        record["comment_generations"] = gen_record[
                            "comment_generations"
                        ]
                        _attach_generation_diffs(record)
                _stamp_record_intents(record, intent_map)
                self._send_json(record)
                return

            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            params = parse_qs(parsed.query)

            if path == "/intents":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    self._send_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
                    return
                entries = body.get("entries") or {}
                changed = _write_intents_to_source(source_path, entries)
                # The source changed on disk; drop the cached view so the next
                # read reflects it. (Other views pick it up on restart.)
                intents_cache["payload"] = None
                saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._send_json({"ok": True, "changed": changed, "saved_at": saved_at})
                return

            if path == "/notes":
                state = self._generation_state(self._resolve_gen_id(params))
                if state is None:
                    self._send_json(
                        {"error": "no generation to attach notes to"},
                        HTTPStatus.BAD_REQUEST,
                    )
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    self._send_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
                    return
                saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                payload = {
                    "run": run_dir.name,
                    "generation": self._resolve_gen_id(params),
                    "saved_at": saved_at,
                    "failure_modes": body.get("failure_modes") or [],
                    "entries": body.get("entries") or {},
                }
                notes_path = state["notes_path"]
                notes_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logging.info(
                    "Saved %d review entries and %d failure modes to %s",
                    len(payload["entries"]),
                    len(payload["failure_modes"]),
                    notes_path,
                )
                self._send_json({"ok": True, "saved_at": saved_at})
                return

            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return DashboardHandler


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def serve(run_dir: Path, port: int | None, open_browser: bool) -> None:
    source_path, base_sidebar, meta = _load_index(run_dir, DATASET_FILENAME)

    # Intent labels live on the source comments, shared across generations. Build
    # the {key -> intent} map once and stamp it onto every sidebar view so the
    # chips are consistent everywhere (rebuilt on restart after a save).
    intent_map = _build_intent_map(source_path)
    _stamp_sidebar_intents(base_sidebar, intent_map)
    intents_cache: dict = {}

    base_ordered = _display_order(base_sidebar)

    # One shared source, many generations. Each generation's sidebar annotations
    # and notes are served on demand and cached per id (see _generation_state).
    generation_list = generate.list_generations(run_dir)
    generations_by_id = {gen["id"]: gen for gen in generation_list}
    default_gen_id = generation_list[0]["id"] if generation_list else None
    states: dict[str, dict] = {}

    meta["has_generations"] = bool(generation_list)
    meta["default_generation"] = default_gen_id
    meta["intent_values"] = list(INTENT_VALUES)
    meta["generations"] = [
        {
            "id": gen["id"],
            "label": gen["label"],
            "created_at": gen["created_at"],
            "model_names": gen["model_names"],
            "config": gen["config"],
        }
        for gen in generation_list
    ]

    html_bytes = DASHBOARD_HTML.read_bytes()
    chosen_port = port if port is not None else _pick_free_port()
    handler_class = _make_handler(
        html_bytes,
        source_path,
        base_sidebar,
        base_ordered,
        meta,
        run_dir,
        generations_by_id,
        default_gen_id,
        states,
        intent_map,
        intents_cache,
    )
    server = ThreadingHTTPServer(("127.0.0.1", chosen_port), handler_class)
    url = f"http://127.0.0.1:{chosen_port}/"

    logging.info("Serving %d records from %s", meta["count"], run_dir)
    logging.info("Dashboard: %s", url)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as error:
            logging.warning("Could not open browser: %s", error)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
    finally:
        server.server_close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview dataset_sample.jsonl in a local HTML dashboard."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Run directory containing dataset_sample.jsonl (defaults to latest run).",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (default: random free port)."
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the dashboard in the browser.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    run_dir = args.run_dir if args.run_dir is not None else latest_dataset_directory()
    serve(run_dir, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(sys.argv[1:])
