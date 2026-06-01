import ast
import io
import tokenize


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
            }
        )

    return docstring_entries


def _collect_comment_entries(file_content: str) -> list[dict]:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    source_lines = file_content.splitlines(keepends=True)
    raw_comment_tokens: list[tuple[int, str, bool]] = []

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        row_idx = row - 1
        if row_idx < 0 or row_idx >= len(source_lines):
            continue
        is_standalone = source_lines[row_idx][:col].strip() == ""
        comment_text = tok.string.rstrip("\r\n")
        raw_comment_tokens.append((row, comment_text, is_standalone))

    if not raw_comment_tokens:
        return []

    raw_comment_tokens.sort(key=lambda token: token[0])

    comment_entries: list[dict] = []
    for line_no, text, is_standalone in raw_comment_tokens:
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
                }
            )

    return comment_entries


def get_comments_from_file(
    file_content: str, include_docstrings: bool = True
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


def strip_comments_from_file(file_content: str, strip_docstrings: bool = True) -> str:
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
