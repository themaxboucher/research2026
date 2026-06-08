import ast
import io
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


def _strip_docstring_lines(source_lines: list[str], file_content: str) -> list[str]:
    tree = ast.parse(file_content)
    stripped_lines = list(source_lines)

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

    return stripped_lines


def strip_comments_from_file(
    file_content: str | None, strip_docstrings: bool = True
) -> str | None:
    if not file_content:
        return file_content

    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    source_lines = file_content.splitlines(keepends=True)

    if strip_docstrings:
        source_lines = _strip_docstring_lines(source_lines, file_content)

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        line = source_lines[row - 1]
        source_lines[row - 1] = line[:col].rstrip() + "\n"

    return "".join(source_lines)
