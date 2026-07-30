import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from collect.comments import extract_comments
from collect.filter_rules import target_comments
from generate.model_output import strip_output_wrappers
from generate.parse_code import enclosing_scope_name, local_scope_bounds, scope_code
from generate.prompt import build_location_prompt
from generate.providers.models import ModelProfile


def _diff_region_bounds(
    source_code: str, source_lines: list[str], anchor_line: int
) -> tuple[int, int]:
    MAX_LINE_COUNT = 500

    qualified_name = enclosing_scope_name(source_code, anchor_line)
    if qualified_name is not None:
        scope_bounds = local_scope_bounds(source_lines, source_code, qualified_name)
        if scope_bounds is not None:
            return scope_bounds

    top = bottom = anchor_line
    while (bottom - top + 1) < min(MAX_LINE_COUNT, len(source_lines)):
        if top > 1:
            top -= 1
        if bottom < len(source_lines):
            bottom += 1
    return top, bottom


_HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _iter_hunks(diff: str):
    current: dict | None = None
    for line in diff.splitlines(keepends=True):
        match = _HUNK_HEADER_PATTERN.match(line)
        if match:
            if current is not None:
                yield current
            # Anything after the closing `@@` is the function-context hint
            # git adds (e.g. " def foo():"). Keep it so the LLM gets the same
            # signal git would have shown.
            header_suffix = line[match.end() :].rstrip("\n")
            current = {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2)) if match.group(2) else 1,
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4)) if match.group(4) else 1,
                "header_suffix": header_suffix,
                "body": [],
            }
        elif current is not None:
            current["body"].append(line)
    if current is not None:
        yield current


def _target_line_code(body_line: str, comment_data: dict) -> str | None:
    if comment_data["type"] != "inline":
        return None
    anchor = comment_data.get("anchor")
    if anchor is not None:
        return anchor
    return body_line[1:].split("#", 1)[0].rstrip()


def _stripped_hunk_body(hunk: dict, comment_data: dict) -> list[str]:
    target_start_line = comment_data["start_line"]
    target_end_line = comment_data["end_line"]
    output_body: list[str] = []
    pending_dashes: list[str] = []  # `-` lines awaiting pairing with `+` lines
    new_line = hunk["new_start"]

    for body_line in hunk["body"]:
        if not body_line or body_line[0] == "\\":
            # Markers like "\ No newline at end of file" stay as-is.
            output_body.append(body_line)
            continue

        prefix = body_line[0]
        if prefix == " ":
            output_body.extend(pending_dashes)
            pending_dashes = []
            output_body.append(body_line)
            new_line += 1
        elif prefix == "-":
            pending_dashes.append(body_line)
        elif prefix == "+":
            line_is_target = target_start_line <= new_line <= target_end_line
            paired_dash = pending_dashes.pop(0) if pending_dashes else None
            if line_is_target:
                code = _target_line_code(body_line, comment_data)
                # Compare the code either side of the change, ignoring comments.
                # A naive split can only over-report a change (a `#` in a string),
                # which is safe: it shows a `-`/`+` pair rather than hiding a change.
                old_code = (
                    None
                    if paired_dash is None
                    else paired_dash[1:].split("#", 1)[0].rstrip()
                )
                code_is_unchanged = (
                    old_code is not None and old_code == (code or "").rstrip()
                )
                if code is None or code_is_unchanged:
                    # Nothing left to show, or the comment was the only change:
                    # fold back to a single context line where one existed.
                    if paired_dash is not None:
                        output_body.append(" " + paired_dash[1:])
                else:
                    if paired_dash is not None:
                        output_body.append(paired_dash)
                    output_body.append("+" + code + "\n")
            else:
                if paired_dash is not None:
                    output_body.append(paired_dash)
                output_body.append(body_line)
            new_line += 1
        else:
            output_body.append(body_line)

    output_body.extend(pending_dashes)
    return output_body


def _scope_diff(diff: str, source_code: str, comment_data: dict) -> str | None:
    source_lines = source_code.splitlines()

    region_start, region_end = _diff_region_bounds(
        source_code, source_lines, comment_data["start_line"]
    )

    overlapping = []
    for hunk in _iter_hunks(diff):
        new_end = hunk["new_start"] + max(hunk["new_count"] - 1, 0)
        overlaps_region = hunk["new_start"] <= region_end and region_start <= new_end
        if overlaps_region:
            overlapping.append(hunk)
    if not overlapping:
        return None
    overlapping.sort(key=lambda hunk: hunk["new_start"])

    # Widen the region so it fully contains every hunk we merge in — a hunk's
    # few lines of git context may spill just past the scope boundary.
    region_start = min(region_start, overlapping[0]["new_start"])
    last_hunk = overlapping[-1]
    last_hunk_end = last_hunk["new_start"] + max(last_hunk["new_count"] - 1, 0)
    region_end = min(max(region_end, last_hunk_end), len(source_lines))

    body: list[str] = []
    new_cursor = region_start  # Next new-side line not yet emitted.
    for hunk in overlapping:
        for line_no in range(new_cursor, hunk["new_start"]):
            body.append(" " + source_lines[line_no - 1] + "\n")
        body.extend(_stripped_hunk_body(hunk, comment_data))
        # The hunk occupies its full committed new-side span even where the
        # target comment was dropped, so advance by the original count to keep
        # the padding from re-emitting those lines.
        new_cursor = hunk["new_start"] + hunk["new_count"]
    for line_no in range(new_cursor, region_end + 1):
        body.append(" " + source_lines[line_no - 1] + "\n")

    old_start = overlapping[0]["old_start"] - (
        overlapping[0]["new_start"] - region_start
    )
    old_count = sum(1 for line in body if line and line[0] in " -")
    new_count = sum(1 for line in body if line and line[0] in " +")
    header = (
        f"@@ -{old_start},{old_count} +{region_start},{new_count} @@"
        f"{overlapping[0].get('header_suffix', '')}\n"
    )
    return header + "".join(body)


def _reverted_comment_text(target: dict, old_comments: list[dict]) -> str | None:
    for old_comment in old_comments:
        if (
            old_comment["type"] == target["type"]
            and old_comment.get("anchor") == target.get("anchor")
            and old_comment.get("comment") != target.get("comment")
        ):
            return old_comment.get("comment")
    return None


def _generate_with_llm(
    prompt: str,
    filepath: str,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict:
    raw_response = None
    try:
        raw_response = get_completion(model_name, prompt)
        comment_text = strip_output_wrappers(raw_response)
        if not comment_text:
            raise ValueError("Model returned an empty comment")
    except Exception as error:
        logging.warning(
            "Failed to generate comment in %s with model %s: %s",
            filepath,
            model_name,
            error,
        )
        return {
            "model": model_name,
            "raw_response": raw_response,
            "comment_text": None,
            "error": str(error),
        }
    return {
        "model": model_name,
        "raw_response": raw_response,
        "comment_text": comment_text,
        "error": None,
    }


def _run_llms_concurrently(
    prompt: str,
    filepath: str,
    model_profile: ModelProfile,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(model_profile.model_names)) as executor:
        futures = [
            executor.submit(
                _generate_with_llm,
                prompt,
                filepath,
                model_name,
                model_profile.get_completion,
            )
            for model_name in model_profile.model_names
        ]
        return [future.result() for future in as_completed(futures)]


def _comment_generation(
    file_data: dict,
    comment_data: dict,
    model_profile: ModelProfile,
) -> dict:

    source_code = file_data["source_code"]
    diff = file_data["diff"]
    previous_source_code = file_data["previous_source_code"]
    filepath = file_data["new_path"]
    intent = comment_data.get("intent")

    if comment_data["status"] == "modified":
        previous_comments = extract_comments(previous_source_code)
        unmodified_comment = _reverted_comment_text(comment_data, previous_comments)
        diff_hunk = _scope_diff(diff, source_code, comment_data)
        prompt = build_location_prompt(
            file_data["repo_name"],
            filepath,
            comment_data,
            commit_message=file_data["commit_message"],
            status="modified",
            intent=intent,
            diff_hunk=diff_hunk,
            unmodified_comment=unmodified_comment,
        )

    if comment_data["status"] == "added":
        code = scope_code(source_code, comment_data)
        prompt = build_location_prompt(
            file_data["repo_name"],
            filepath,
            comment_data,
            commit_message=file_data["commit_message"],
            status="added",
            intent=intent,
            scope_code=code,
        )

    results = _run_llms_concurrently(prompt, filepath, model_profile)
    return {
        "type": comment_data["type"],
        "status": comment_data["status"],
        "start_line": comment_data["start_line"],
        "end_line": comment_data["end_line"],
        "anchor": comment_data.get("anchor"),
        "comment": comment_data.get("comment"),
        "prompt": prompt,
        "results": results,
    }


def location_generate_for_file(file_data: dict, model_profile: ModelProfile) -> dict:
    comment_generations = []
    for comment_data in target_comments(file_data):
        try:
            comment_generations.append(
                _comment_generation(file_data, comment_data, model_profile)
            )
        except Exception as error:
            logging.warning(
                "Skipping a comment in %s: could not build generation inputs: %s",
                file_data.get("new_path"),
                error,
            )

    return {
        "repo_name": file_data["repo_name"],
        "commit_hash": file_data.get("commit_hash"),
        "new_path": file_data["new_path"],
        "comment_generations": comment_generations,
    }
