import re
from typing import NamedTuple

NO_EDITS_MARKER = "NO_EDITS"

# git-style conflict markers, matching the aider "diff" edit format:
# https://aider.chat/docs/more/edit-formats.html
_SEARCH_MARKER = re.compile(r"^<{5,9} SEARCH\s*$")
_DIVIDER_MARKER = re.compile(r"^={5,9}\s*$")
_REPLACE_MARKER = re.compile(r"^>{5,9} REPLACE\s*$")


class SearchReplaceEdit(NamedTuple):
    search: str
    replace: str


def parse_edit_response(llm_response: str) -> list[SearchReplaceEdit]:
    if llm_response.strip() == NO_EDITS_MARKER:
        return []
    edits = _parse_search_replace_blocks(llm_response)
    if not edits:
        raise ValueError(
            f"Response contains neither SEARCH/REPLACE blocks nor {NO_EDITS_MARKER}"
        )
    return edits


def _parse_search_replace_blocks(text: str) -> list[SearchReplaceEdit]:
    lines = text.splitlines()
    edits: list[SearchReplaceEdit] = []
    index = 0
    while index < len(lines):
        if not _SEARCH_MARKER.match(lines[index]):
            index += 1
            continue
        search_lines, index = _collect_until(lines, index + 1, _DIVIDER_MARKER)
        replace_lines, index = _collect_until(lines, index, _REPLACE_MARKER)
        edits.append(
            SearchReplaceEdit("\n".join(search_lines), "\n".join(replace_lines))
        )
    return edits


def _collect_until(
    lines: list[str], index: int, terminator: re.Pattern
) -> tuple[list[str], int]:
    collected = []
    while index < len(lines) and not terminator.match(lines[index]):
        if _SEARCH_MARKER.match(lines[index]) or _REPLACE_MARKER.match(lines[index]):
            raise ValueError("Malformed SEARCH/REPLACE block: missing a divider")
        collected.append(lines[index])
        index += 1
    if index >= len(lines):
        raise ValueError("Malformed SEARCH/REPLACE block: missing a closing marker")
    return collected, index + 1


# Models rarely copy the SEARCH section byte-for-byte, so matching falls back
# through progressively looser line comparisons: exact, then ignoring trailing
# whitespace, then ignoring indentation entirely.
_MATCH_NORMALIZERS = (
    lambda line: line,
    lambda line: line.rstrip(),
    lambda line: line.strip(),
)


def apply_edits(file_content: str, edits: list[SearchReplaceEdit]) -> str:
    file_lines = file_content.splitlines()
    for edit in edits:
        file_lines = _apply_single_edit(file_lines, edit)
    return "\n".join(file_lines) + "\n"


def _apply_single_edit(
    file_lines: list[str], edit: SearchReplaceEdit
) -> list[str]:
    search_lines = edit.search.splitlines()
    replace_lines = edit.replace.splitlines()

    if not search_lines:
        raise ValueError("SEARCH section is empty")

    match_start = _find_unique_match(file_lines, search_lines)
    window_size = len(search_lines)
    matched_file_lines = file_lines[match_start : match_start + window_size]
    indent_prefix = _indent_difference(matched_file_lines, search_lines)
    adjusted_replace = _reindent(replace_lines, indent_prefix)
    return (
        file_lines[:match_start]
        + adjusted_replace
        + file_lines[match_start + window_size :]
    )


def _find_unique_match(file_lines: list[str], search_lines: list[str]) -> int:
    window_size = len(search_lines)
    for normalize in _MATCH_NORMALIZERS:
        target = [normalize(line) for line in search_lines]
        match_starts = [
            start
            for start in range(len(file_lines) - window_size + 1)
            if [normalize(line) for line in file_lines[start : start + window_size]]
            == target
        ]
        if len(match_starts) > 1:
            raise ValueError("SEARCH section matches multiple locations in the file")
        if match_starts:
            return match_starts[0]
    raise ValueError("SEARCH section does not match any location in the file")


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _indent_difference(
    matched_file_lines: list[str], search_lines: list[str]
) -> str:
    """How much extra indentation the file has over the model's SEARCH lines.

    Returned as a whitespace prefix to prepend to the REPLACE lines so they land
    at the file's indentation even when the model dedented the whole block.
    """
    for file_line, search_line in zip(matched_file_lines, search_lines):
        if not search_line.strip():
            continue
        file_indent = _leading_whitespace(file_line)
        search_indent = _leading_whitespace(search_line)
        if file_indent.endswith(search_indent):
            return file_indent[: len(file_indent) - len(search_indent)]
        return ""
    return ""


def _reindent(lines: list[str], prefix: str) -> list[str]:
    if not prefix:
        return lines
    return [prefix + line if line.strip() else line for line in lines]
