import io
import logging
import textwrap
import tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

from generate.validate import target_comments
from generate.model_output import strip_code_output_wrappers
from generate.prompt import build_regenerate_prompt
from generate.parse_code import scope_bounds


class CodeAnalysis(NamedTuple):
    code_lines: list[tuple[int, str]]
    standalone_comments: dict[int, str]
    inline_comments: dict[int, str]


def analyze_code(text: str) -> CodeAnalysis:
    """Tokenize `text` (dedented so indented scopes parse) and separate its
    code lines from its comments. Raises on code that cannot be tokenized —
    for model output that means a truncated or malformed regeneration."""
    dedented = textwrap.dedent(text)
    if not dedented.endswith("\n"):
        dedented += "\n"
    lines = dedented.splitlines()

    comment_columns: dict[int, int] = {}
    standalone_comments: dict[int, str] = {}
    inline_comments: dict[int, str] = {}
    tokens = tokenize.tokenize(io.BytesIO(dedented.encode("utf-8")).readline)
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        line_index = row - 1
        comment_columns[line_index] = column
        comment_text = token.string.rstrip()
        line_is_only_comment = lines[line_index][:column].strip() == ""
        if line_is_only_comment:
            standalone_comments[line_index] = comment_text
        else:
            inline_comments[line_index] = comment_text

    code_lines: list[tuple[int, str]] = []
    for line_index, line in enumerate(lines):
        if line_index in comment_columns:
            line = line[: comment_columns[line_index]]
        code = line.rstrip()
        if code:
            code_lines.append((line_index, code))

    return CodeAnalysis(code_lines, standalone_comments, inline_comments)


def _covering_target(line_no: int, targets: list[dict]) -> dict | None:
    for target in targets:
        if target["start_line"] <= line_no <= target["end_line"]:
            return target
    return None


def stripped_scope_lines(
    source_lines: list[str],
    scope_start_line: int,
    scope_end_line: int,
    targets_to_strip: list[dict],
) -> list[tuple[int, str]]:
    """The scope's lines with every target comment removed, as
    (original line number, line text) pairs. Block target lines are dropped
    entirely. An inline target keeps its code with the comment removed."""
    kept_lines: list[tuple[int, str]] = []
    for line_no in range(scope_start_line, scope_end_line + 1):
        line = source_lines[line_no - 1]
        target = _covering_target(line_no, targets_to_strip)
        if target is None:
            kept_lines.append((line_no, line))
            continue
        if target["type"] != "inline":
            continue
        anchor = target.get("anchor")
        code = anchor if anchor is not None else line.split("#", 1)[0].rstrip()
        if code:
            kept_lines.append((line_no, code))
    return kept_lines


def _target_positions(
    kept_lines: list[tuple[int, str]],
    input_analysis: CodeAnalysis,
    targets: list[dict],
) -> list[int | None]:
    """For each target, its anchor's position in the input's code-line
    sequence: an inline target anchors to its own code line, a block target to
    the first code line after it. None when the anchor isn't in the scope
    (e.g. a trailing block comment), which makes the target unscorable."""
    original_line_numbers = [line_no for line_no, _ in kept_lines]

    positions: list[int | None] = []
    for target in targets:
        position = None
        for code_position, (line_index, _) in enumerate(input_analysis.code_lines):
            original_line_no = original_line_numbers[line_index]
            if target["type"] == "inline":
                anchor_found = original_line_no == target["start_line"]
            else:
                anchor_found = original_line_no > target["end_line"]
            if anchor_found:
                position = code_position
                break
        positions.append(position)
    return positions


def _code_sequence(analysis: CodeAnalysis) -> list[str]:
    return [code for _, code in analysis.code_lines]


def _code_mismatch_error(input_codes: list[str], output_codes: list[str]) -> str | None:
    if input_codes == output_codes:
        return None
    for position, (expected, actual) in enumerate(zip(input_codes, output_codes)):
        if expected != actual:
            return (
                f"regenerated code line {position + 1} differs from the "
                f"original: expected {expected!r}, got {actual!r}"
            )
    return (
        f"regenerated code line count differs from the original: "
        f"expected {len(input_codes)} code lines, got {len(output_codes)}"
    )


def _block_prediction(
    output_analysis: CodeAnalysis, anchor_line_index: int, given_comment_texts: set[str]
) -> str | None:
    """The comment block sitting contiguously above the anchor line in the
    regenerated code, with any lines the model was already given subtracted so
    preserved pre-existing comments aren't scored as predictions."""
    block_lines: list[str] = []
    line_index = anchor_line_index - 1
    while line_index >= 0 and line_index in output_analysis.standalone_comments:
        block_lines.append(output_analysis.standalone_comments[line_index])
        line_index -= 1
    block_lines.reverse()

    new_lines = [
        line for line in block_lines if line.strip() not in given_comment_texts
    ]
    if not new_lines:
        return None
    return "\n".join(new_lines)


def _inline_prediction(
    output_analysis: CodeAnalysis, anchor_line_index: int, given_comment_texts: set[str]
) -> str | None:
    comment_text = output_analysis.inline_comments.get(anchor_line_index)
    if comment_text is None or comment_text.strip() in given_comment_texts:
        return None
    return comment_text


def _extraction_for_target(
    target: dict,
    anchor_position: int | None,
    output_analysis: CodeAnalysis,
    given_comment_texts: set[str],
) -> dict:
    if anchor_position is None:
        return {
            "comment_text": None,
            "placement_hit": None,
            "form_matches": None,
            "error": "target has no anchor code line within the scope",
        }

    anchor_line_index, _ = output_analysis.code_lines[anchor_position]
    predictions_by_form = {
        "block": _block_prediction(
            output_analysis, anchor_line_index, given_comment_texts
        ),
        "inline": _inline_prediction(
            output_analysis, anchor_line_index, given_comment_texts
        ),
    }

    # Either form at the right spot counts as a hit; prefer the form the human
    # used when the model produced both.
    preferred_form = target["type"]
    other_form = "inline" if preferred_form == "block" else "block"
    for form in (preferred_form, other_form):
        prediction = predictions_by_form[form]
        if prediction is not None:
            return {
                "comment_text": prediction,
                "placement_hit": True,
                "form_matches": form == target["type"],
                "error": None,
            }

    return {
        "comment_text": None,
        "placement_hit": False,
        "form_matches": None,
        "error": None,
    }


def _regenerate_with_model(
    prompt: str,
    input_analysis: CodeAnalysis,
    anchor_positions: list[int | None],
    targets: list[dict],
    given_comment_texts: set[str],
    model_name: str,
    get_completion,
) -> dict:
    raw_response = get_completion(model_name, prompt)
    try:
        regenerated_code = strip_code_output_wrappers(raw_response)
        if not regenerated_code:
            raise ValueError("Model returned an empty response")
        output_analysis = analyze_code(regenerated_code)
        mismatch_error = _code_mismatch_error(
            _code_sequence(input_analysis), _code_sequence(output_analysis)
        )
        if mismatch_error is not None:
            raise ValueError(mismatch_error)
    except Exception as error:
        return {
            "model": model_name,
            "raw_response": raw_response,
            "regenerated_code": None,
            "error": str(error),
            "extractions": None,
        }

    extractions = [
        _extraction_for_target(target, position, output_analysis, given_comment_texts)
        for target, position in zip(targets, anchor_positions)
    ]
    return {
        "model": model_name,
        "raw_response": raw_response,
        "regenerated_code": regenerated_code,
        "error": None,
        "extractions": extractions,
    }


def _run_llms_concurrently(
    prompt: str,
    input_analysis: CodeAnalysis,
    anchor_positions: list[int | None],
    targets: list[dict],
    given_comment_texts: set[str],
    model_profile,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(model_profile.model_names)) as executor:
        futures = [
            executor.submit(
                _regenerate_with_model,
                prompt,
                input_analysis,
                anchor_positions,
                targets,
                given_comment_texts,
                model_name,
                model_profile.get_completion,
            )
            for model_name in model_profile.model_names
        ]
        return [future.result() for future in as_completed(futures)]


def _scope_regeneration(
    file_data: dict, source_code: str, scope_group: dict, model_profile
) -> dict:
    source_lines = source_code.splitlines()
    kept_lines = stripped_scope_lines(
        source_lines,
        scope_group["start_line"],
        scope_group["end_line"],
        scope_group["targets_to_strip"],
    )
    input_code = "\n".join(text for _, text in kept_lines)
    input_analysis = analyze_code(input_code)
    anchor_positions = _target_positions(
        kept_lines, input_analysis, scope_group["targets"]
    )
    given_comment_texts = {
        comment.strip()
        for comment in (
            *input_analysis.standalone_comments.values(),
            *input_analysis.inline_comments.values(),
        )
    }

    prompt = build_regenerate_prompt(
        file_data["repo_name"],
        file_data["new_path"],
        file_data["commit_message"],
        input_code,
    )
    results = _run_llms_concurrently(
        prompt,
        input_analysis,
        anchor_positions,
        scope_group["targets"],
        given_comment_texts,
        model_profile,
    )

    return {
        "repo_name": file_data["repo_name"],
        "commit_hash": file_data.get("commit_hash"),
        "new_path": file_data["new_path"],
        "scope_start_line": scope_group["start_line"],
        "scope_end_line": scope_group["end_line"],
        "input_code": input_code,
        "targets": [
            {
                "start_line": target["start_line"],
                "end_line": target["end_line"],
                "type": target["type"],
                "anchor": target.get("anchor"),
                "intent": target.get("intent"),
                "comment": target.get("comment"),
            }
            for target in scope_group["targets"]
        ],
        "prompt": prompt,
        "results": results,
    }


def _group_targets_by_scope(source_code: str, target_comments: list[dict]) -> list[dict]:
    source_lines = source_code.splitlines()

    targets_by_bounds: dict[tuple[int, int], list[dict]] = {}
    for target in target_comments:
        bounds = scope_bounds(source_code, source_lines, target["start_line"])
        targets_by_bounds.setdefault(bounds, []).append(target)

    scope_groups = []
    for (start_line, end_line), scoped_targets in sorted(targets_by_bounds.items()):
        targets_to_strip = [
            target
            for target in target_comments
            if start_line <= target["start_line"] and target["end_line"] <= end_line
        ]
        scope_groups.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "targets": scoped_targets,
                "targets_to_strip": targets_to_strip,
            }
        )
    return scope_groups


def regenerate_generate_for_file(file_data: dict, model_profile) -> list[dict]:
    source_code = file_data["source_code"]

    scope_records = []
    targets = target_comments(file_data)

    # Multiple target comments may sit in the same local scope. We group them so that when an LLM regenerates
    # the scope we can compare the added comments against all the target comments in that scope.
    for scope_group in _group_targets_by_scope(source_code, targets):
        try:
            scope_records.append(
                _scope_regeneration(file_data, source_code, scope_group, model_profile)
            )
        except Exception as error:
            logging.warning(
                "Skipping a scope in %s: could not build regeneration inputs: %s",
                file_data.get("new_path"),
                error,
            )
    return scope_records
