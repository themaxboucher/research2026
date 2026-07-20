import ast
import io
import textwrap
import tokenize


def _node_signature(node: ast.AST) -> str:
    if isinstance(node, ast.Module):
        return "<module>"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def {node.name}({ast.unparse(node.args)})"
    if isinstance(node, ast.ClassDef):
        base_sources = ", ".join(ast.unparse(base) for base in node.bases)
        return (
            f"class {node.name}({base_sources})"
            if base_sources
            else f"class {node.name}"
        )
    return ""


def _next_code_line(source_lines: list[str], after_line_no: int) -> str | None:
    for line_index in range(after_line_no, len(source_lines)):
        stripped_line = source_lines[line_index].strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        return stripped_line
    return None


def _collect_docstring_entries(file_content: str) -> list[dict]:
    tree = ast.parse(file_content)
    docstring_entries = []

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if not node.body:
            continue
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            continue
        if not isinstance(first_stmt.value, ast.Constant) or not isinstance(
            first_stmt.value.value, str
        ):
            continue

        raw_text = ast.get_source_segment(file_content, first_stmt) or ""
        docstring_entries.append(
            {
                "comment": raw_text,
                "type": "docstring",
                "start_line": first_stmt.lineno,
                "end_line": first_stmt.end_lineno,
                "anchor": _node_signature(node),
            }
        )

    return docstring_entries


def _collect_comment_entries(file_content: str) -> list[dict]:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    source_lines = file_content.splitlines(keepends=True)
    raw_comment_tokens: list[tuple[int, str, bool, str | None]] = []

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        row_idx = row - 1
        if row_idx < 0 or row_idx >= len(source_lines):
            continue
        prefix = source_lines[row_idx][:col]
        is_standalone = prefix.strip() == ""
        inline_anchor = None if is_standalone else prefix.rstrip()
        comment_text = tok.string.rstrip("\r\n")
        raw_comment_tokens.append((row, comment_text, is_standalone, inline_anchor))

    if not raw_comment_tokens:
        return []

    raw_comment_tokens.sort(key=lambda token: token[0])

    comment_entries: list[dict] = []
    for line_no, text, is_standalone, inline_anchor in raw_comment_tokens:
        comment_type = "block" if is_standalone else "inline"
        previous_entry = comment_entries[-1] if comment_entries else None

        is_block_continuation = (
            previous_entry is not None
            and is_standalone
            and previous_entry["type"] == "block"
            and previous_entry["end_line"] + 1 == line_no
        )

        if is_block_continuation:
            previous_entry["comment"] += "\n" + text
            previous_entry["end_line"] = line_no
        else:
            comment_entries.append(
                {
                    "comment": text,
                    "type": comment_type,
                    "start_line": line_no,
                    "end_line": line_no,
                    "anchor": inline_anchor,
                }
            )

    plain_source_lines = file_content.splitlines()
    for entry in comment_entries:
        if entry["type"] == "block":
            entry["anchor"] = _next_code_line(plain_source_lines, entry["end_line"])

    return comment_entries


def extract_comments(
    file_content: str | None, include_docstrings: bool = True
) -> list[dict]:
    if not file_content:
        return []

    comment_entries = _collect_comment_entries(file_content)

    if not include_docstrings:
        return comment_entries

    docstring_entries = _collect_docstring_entries(file_content)
    all_entries = comment_entries + docstring_entries
    all_entries.sort(key=lambda entry: entry["start_line"])

    return all_entries


def _is_same_comment(first_comment: dict, second_comment: dict) -> bool:
    return first_comment.get("comment") == second_comment.get("comment")


def _is_modified_comment(first_comment: dict, second_comment: dict) -> bool:
    first_anchor = first_comment.get("anchor")
    second_anchor = second_comment.get("anchor")
    if first_anchor is None or second_anchor is None:
        return False
    return (
        first_anchor == second_anchor
        and first_comment["type"] == second_comment["type"]
    )


def _find_status_for_new_comment(
    new_comment: dict,
    old_comments: list[dict],
    matched_old_comment_indices: set[int],
) -> tuple[str, int | None]:
    for old_index, old_comment in enumerate(old_comments):
        if old_index in matched_old_comment_indices:
            continue
        if _is_same_comment(new_comment, old_comment):
            return "unchanged", old_index

    for old_index, old_comment in enumerate(old_comments):
        if old_index in matched_old_comment_indices:
            continue
        if _is_modified_comment(new_comment, old_comment):
            return "modified", old_index

    return "added", None


def _annotate_comments(
    new_comments: list[dict], old_comments: list[dict]
) -> list[dict]:
    if not old_comments:
        return [{**comment, "status": "added"} for comment in new_comments]

    if not new_comments:
        return [{**comment, "status": "removed"} for comment in old_comments]

    matched_old_comment_indices: set[int] = set()
    annotated_comments: list[dict] = []

    for new_comment in new_comments:
        status, matched_old_index = _find_status_for_new_comment(
            new_comment, old_comments, matched_old_comment_indices
        )
        if matched_old_index is not None:
            matched_old_comment_indices.add(matched_old_index)
        annotated_comments.append({**new_comment, "status": status})

    for old_index, old_comment in enumerate(old_comments):
        if old_index in matched_old_comment_indices:
            continue
        annotated_comments.append({**old_comment, "status": "removed"})

    annotated_comments.sort(key=lambda comment: comment["start_line"])
    return annotated_comments


def get_comments_from_file(file_content: str, previous_file_content: str) -> list[dict]:
    new_comments = extract_comments(file_content)
    old_comments = extract_comments(previous_file_content)
    return _annotate_comments(new_comments, old_comments)


def _line_indentation(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _graft_inline_comment(
    target_lines: list[str], comment_entry: dict, claimed_line_indices: set[int]
) -> None:
    code_anchor = comment_entry["anchor"]
    for line_index, line in enumerate(target_lines):
        if line_index in claimed_line_indices:
            continue
        if line.rstrip() != code_anchor:
            continue
        target_lines[line_index] = code_anchor + "  " + comment_entry["comment"]
        claimed_line_indices.add(line_index)
        return


def _block_comment_insertion(
    target_lines: list[str], comment_entry: dict, claimed_line_indices: set[int]
) -> tuple[int, list[str]] | None:
    code_anchor = comment_entry["anchor"]
    if code_anchor is None:
        return None
    for line_index, line in enumerate(target_lines):
        if line_index in claimed_line_indices:
            continue
        if line.strip() != code_anchor:
            continue
        claimed_line_indices.add(line_index)
        indentation = _line_indentation(line)
        comment_lines = [
            indentation + comment_line
            for comment_line in comment_entry["comment"].splitlines()
        ]
        return line_index, comment_lines
    return None


def _docstring_insert_points(source: str) -> dict[str, list[tuple[int, str]]]:
    tree = ast.parse(source)
    source_lines = source.splitlines()
    insert_points: dict[str, list[tuple[int, str]]] = {}

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if not node.body:
            continue
        first_statement_line_index = node.body[0].lineno - 1
        indentation = _line_indentation(source_lines[first_statement_line_index])
        insert_points.setdefault(_node_signature(node), []).append(
            (first_statement_line_index, indentation)
        )

    return insert_points


def _docstring_insertion(
    insert_points: dict[str, list[tuple[int, str]]], comment_entry: dict
) -> tuple[int, list[str]] | None:
    available_points = insert_points.get(comment_entry["anchor"])
    if not available_points:
        return None
    line_index, indentation = available_points.pop(0)
    docstring_lines = comment_entry["comment"].splitlines()
    rendered_lines = [indentation + docstring_lines[0]] + docstring_lines[1:]
    return line_index, rendered_lines


def graft_comments_onto_code(
    code_without_comments: str, previous_file_content: str | None
) -> str:
    previous_comments = extract_comments(previous_file_content)
    if not previous_comments:
        return code_without_comments

    target_lines = code_without_comments.splitlines()
    docstring_points = _docstring_insert_points(code_without_comments)
    claimed_inline_line_indices: set[int] = set()
    claimed_block_line_indices: set[int] = set()
    pending_insertions: list[tuple[int, list[str]]] = []

    for comment_entry in previous_comments:
        if comment_entry["type"] == "inline":
            _graft_inline_comment(
                target_lines, comment_entry, claimed_inline_line_indices
            )
        elif comment_entry["type"] == "block":
            insertion = _block_comment_insertion(
                target_lines, comment_entry, claimed_block_line_indices
            )
            if insertion:
                pending_insertions.append(insertion)
        else:
            insertion = _docstring_insertion(docstring_points, comment_entry)
            if insertion:
                pending_insertions.append(insertion)

    for line_index, lines_to_insert in sorted(pending_insertions, reverse=True):
        target_lines[line_index:line_index] = lines_to_insert

    return "\n".join(target_lines) + "\n"


def _strip_docstring_lines(
    source_lines: list[str], file_content: str
) -> tuple[list[str], set[int]]:
    tree = ast.parse(file_content)
    stripped_lines = list(source_lines)
    docstring_line_indices: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if not node.body:
            continue
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr):
            continue
        if not isinstance(first_stmt.value, ast.Constant) or not isinstance(
            first_stmt.value.value, str
        ):
            continue

        for line_idx in range(first_stmt.lineno - 1, first_stmt.end_lineno):
            stripped_lines[line_idx] = "\n"
            docstring_line_indices.add(line_idx)

    return stripped_lines, docstring_line_indices


def strip_comments_from_file(
    file_content: str | None, strip_docstrings: bool = True
) -> str | None:
    if not file_content:
        return file_content

    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    source_lines = file_content.splitlines(keepends=True)
    lines_to_drop: set[int] = set()

    if strip_docstrings:
        source_lines, docstring_line_indices = _strip_docstring_lines(
            source_lines, file_content
        )
        lines_to_drop.update(docstring_line_indices)

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        line_index = row - 1
        line = source_lines[line_index]
        if line[:col].strip() == "":
            lines_to_drop.add(line_index)
        else:
            source_lines[line_index] = line[:col].rstrip() + "\n"

    return "".join(
        line for index, line in enumerate(source_lines) if index not in lines_to_drop
    )


_MAX_COMMENT_LINE_LENGTH = 80


def _wrap_comment_line(line: str) -> list[str]:
    if len(line) <= _MAX_COMMENT_LINE_LENGTH:
        return [line]
    hash_count = len(line) - len(line.lstrip("#"))
    prefix = line[:hash_count] + " "
    content = line[hash_count:].strip()
    wrapped = textwrap.wrap(
        content,
        width=_MAX_COMMENT_LINE_LENGTH,
        initial_indent=prefix,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or [line]


def _normalize_block_comment(text: str) -> list[str]:
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line.startswith("#")]
    lines = lines or ["# " + text.strip().lstrip("#").strip()]
    return [wrapped for line in lines for wrapped in _wrap_comment_line(line)]


def _normalize_inline_comment(text: str) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not line.startswith("#"):
        line = "# " + line.lstrip("#").strip()
    return line


def _replace_inline_comment(line: str, anchor: str | None, comment_text: str) -> str:
    naive_anchor_fallback = line.split("#", 1)[0].rstrip()
    anchor = anchor or naive_anchor_fallback
    two_spaces = "  "
    return anchor + two_spaces + _normalize_inline_comment(comment_text)


def apply_generated_comment(
    source_code: str, human_comment_data: dict, generated_comment_text: str
) -> str:
    """The source file with the human's target comment replaced by a model's
    generated text, normalized to the comment's form (inline or block)."""
    source_code_lines = source_code.split("\n")
    start_index = human_comment_data["start_line"] - 1

    is_inline_comment = human_comment_data["type"] == "inline"
    if is_inline_comment:
        source_code_lines[start_index] = _replace_inline_comment(
            source_code_lines[start_index],
            human_comment_data.get("anchor"),
            generated_comment_text,
        )
        return "\n".join(source_code_lines)

    indentation = _line_indentation(source_code_lines[start_index])
    block_lines = [
        indentation + comment_line
        for comment_line in _normalize_block_comment(generated_comment_text)
    ]
    source_code_lines[start_index : human_comment_data["end_line"]] = block_lines
    return "\n".join(source_code_lines)
