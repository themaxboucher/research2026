import ast
import io
import tokenize


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
