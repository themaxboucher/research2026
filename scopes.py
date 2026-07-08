import ast

MAX_SCOPE_LINE_COUNT = 500


def ast_nodes(source: str) -> list[tuple[ast.AST, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    nodes: list[tuple[ast.AST, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qualified_name = prefix + child.name
                nodes.append((child, qualified_name))
                visit(child, qualified_name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return nodes


def node_line_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    # Include decorator lines in the span, if any
    for decorator in getattr(node, "decorator_list", []):
        start = min(start, decorator.lineno)
    return start, node.end_lineno or node.lineno


def enclosing_scope_name(source: str, line: int) -> str | None:
    enclosing_scopes = []  # List of scopes enclosing the given line, ordered by span length
    for node, qualified_name in ast_nodes(source):
        start, end = node_line_span(node)
        line_is_within_span = start <= line <= end
        if line_is_within_span:
            span_length = end - start
            enclosing_scopes.append((span_length, qualified_name))
    if not enclosing_scopes:
        return None
    _, innermost_name = min(enclosing_scopes, key=lambda scope: scope[0])
    return innermost_name


def local_scope_bounds(
    source_lines: list[str], source_code: str, qualified_name: str
) -> tuple[int, int] | None:
    for node, name in ast_nodes(source_code):
        if name != qualified_name:
            continue

        # node_line_span already extends the start up to cover decorators.
        start, end = node_line_span(node)

        # Pull in any block comment lines sitting directly above the scope.
        while start > 1 and source_lines[start - 2].strip().startswith("#"):
            start -= 1

        return start, end
    return None


def _scope_bounds(source_code: str, source_lines: list[str], anchor_line: int) -> tuple[int, int]:
    """Line bounds (1-indexed, inclusive) of the local scope enclosing the
    comment. Falls back to the whole module (capped) at module level."""
    qualified_name = enclosing_scope_name(source_code, anchor_line)
    scope_bounds = (
        local_scope_bounds(source_lines, source_code, qualified_name)
        if qualified_name is not None
        else None
    )
    if scope_bounds is not None:
        return scope_bounds
    return 1, min(len(source_lines), MAX_SCOPE_LINE_COUNT)


def scope_code(source_code: str, comment_data: dict) -> str:
    """Return the source of the local scope enclosing the target comment, with
    the target comment itself removed. A block target is dropped entirely; an
    inline target keeps its code and loses only the trailing comment. When the
    comment lives at module level, fall back to the whole module (capped)."""
    source_lines = source_code.splitlines()
    start, end = _scope_bounds(source_code, source_lines, comment_data["start_line"])

    target_start = comment_data["start_line"]
    target_end = comment_data["end_line"]

    output_lines: list[str] = []
    for line_no in range(start, end + 1):
        line = source_lines[line_no - 1]
        line_is_target = target_start <= line_no <= target_end
        if not line_is_target:
            output_lines.append(line)
            continue
        if comment_data["type"] == "inline":
            # Keep the code the inline comment sat on, dropping the comment.
            anchor = comment_data.get("anchor")
            code = anchor if anchor is not None else line.split("#", 1)[0].rstrip()
            if code:
                output_lines.append(code)
        # A block target is wholly comment, so its lines vanish entirely.

    return "\n".join(output_lines)


def scope_code_with_comment(source_code: str, comment_data: dict) -> tuple[str, int, int]:
    """Return the enclosing scope's source with the target comment kept in
    place, plus the comment's 1-indexed line span *within the returned scope*
    (for highlighting). Unlike scope_code, nothing is stripped — the labeler
    needs to see the comment to judge its intent."""
    source_lines = source_code.splitlines()
    start, end = _scope_bounds(source_code, source_lines, comment_data["start_line"])

    target_start = comment_data["start_line"]
    target_end = comment_data["end_line"]

    # The scope lookup can miss the comment — a trailing block comment sitting
    # past the scope's AST end line, or a duplicate qualified name resolving to
    # the wrong node. The comment must always be visible, so widen to include
    # it, falling back to a small window when that union would be unwieldy.
    start = min(start, target_start)
    end = max(end, target_end)
    if end - start + 1 > MAX_SCOPE_LINE_COUNT:
        start = max(1, target_start - 20)
        end = min(len(source_lines), target_end + 20)

    scope_lines = source_lines[start - 1 : end]
    comment_start = target_start - start + 1
    comment_end = target_end - start + 1
    return "\n".join(scope_lines), comment_start, comment_end
