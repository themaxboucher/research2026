import ast

MAX_SCOPE_LINE_COUNT = 500


def source_lines_of(source_code: str) -> list[str]:
    """Lines numbered the way the Python parser numbers them. str.splitlines()
    also breaks on form feeds and unicode separators, both legal in source, and
    every line after one of those would then be off by one."""
    normalized = source_code.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def scope_nodes(source_code: str) -> list[ast.AST]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def node_line_span(node: ast.AST) -> tuple[int, int]:
    start = node.lineno
    # Include decorator lines in the span, if any
    for decorator in getattr(node, "decorator_list", []):
        start = min(start, decorator.lineno)
    return start, node.end_lineno or node.lineno


def local_scope_bounds(
    source_code: str, source_lines: list[str], line: int
) -> tuple[int, int] | None:
    enclosing_scopes = []  # Scopes enclosing the given line, ordered by span length
    for node in scope_nodes(source_code):
        start, end = node_line_span(node)
        line_is_within_span = start <= line <= end
        if line_is_within_span:
            span_length = end - start
            enclosing_scopes.append((span_length, node))
    if not enclosing_scopes:
        return None

    _, innermost_node = min(enclosing_scopes, key=lambda scope: scope[0])

    # node_line_span already extends the start up to cover decorators.
    start, end = node_line_span(innermost_node)

    # Pull in any block comment lines sitting directly above the scope.
    while start > 1 and source_lines[start - 2].strip().startswith("#"):
        start -= 1

    return start, end


def _top_level_spans(source_code: str, source_lines: list[str]) -> list[tuple[int, int]]:
    """Line span of every top-level statement, each extended up over the block
    comment lines sitting directly above it."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as error:
        raise ValueError(f"source code is not parsable: {error}") from error
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for statement in tree.body:
        start, end = node_line_span(statement)
        while start - 1 > previous_end and source_lines[start - 2].strip().startswith("#"):
            start -= 1
        spans.append((start, end))
        previous_end = end
    return spans


def _window_around(
    source_code: str, source_lines: list[str], target_start: int, target_end: int
) -> tuple[int, int]:
    """Window of whole top-level statements around the target lines. Snapping to
    statement boundaries keeps the extracted block parsable on its own, which a
    raw line window is not: it can start midway through an indented block or cut
    a multi-line string in half. A single statement longer than
    MAX_SCOPE_LINE_COUNT is still returned whole, since truncating it would
    break exactly that guarantee."""
    spans = _top_level_spans(source_code, source_lines)
    if not spans:
        return 1, len(source_lines)

    covering = [
        index
        for index, (start, end) in enumerate(spans)
        if start <= target_end and end >= target_start
    ]
    if covering:
        first_index, last_index = covering[0], covering[-1]
    else:
        # The comment sits in the gap between two statements. Take both, so a
        # comment dangling at the end of an indented block keeps its header.
        following = [index for index, (start, _) in enumerate(spans) if start > target_end]
        next_index = following[0] if following else len(spans)
        first_index = max(0, next_index - 1)
        last_index = min(next_index, len(spans) - 1)

    start = min(spans[first_index][0], target_start)
    end = max(spans[last_index][1], target_end)

    while True:
        previous_index = first_index - 1
        next_index = last_index + 1
        grow_up = (
            previous_index >= 0
            and end - spans[previous_index][0] + 1 <= MAX_SCOPE_LINE_COUNT
        )
        grow_down = (
            next_index < len(spans)
            and spans[next_index][1] - start + 1 <= MAX_SCOPE_LINE_COUNT
        )
        if grow_up and grow_down:
            # Keep the comment roughly centred by growing the shorter side.
            grow_up = (target_start - start) <= (end - target_end)
            grow_down = not grow_up
        if grow_up:
            first_index = previous_index
            start = spans[first_index][0]
        elif grow_down:
            last_index = next_index
            end = spans[last_index][1]
        else:
            return start, end


def _scope_bounds(
    source_code: str, source_lines: list[str], target_start: int, target_end: int
) -> tuple[int, int]:
    """Line bounds (1-indexed, inclusive) of the local scope enclosing the
    comment. Falls back to a capped window around the comment at module level."""
    local_bounds = local_scope_bounds(source_code, source_lines, target_start)
    if local_bounds is not None:
        return local_bounds
    return _window_around(source_code, source_lines, target_start, target_end)


def _strip_enclosing_indent(lines: list[str]) -> list[str]:
    """Drop the indentation of the enclosing scope, so a block lifted out of a
    class or function parses on its own. Keyed off the first line rather than
    the common prefix textwrap.dedent looks for: a docstring with a flush-left
    line inside it makes that common prefix empty and dedents nothing."""
    first_line = lines[0]
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    if not indent:
        return lines
    return [
        line[len(indent) :] if line.startswith(indent) else line for line in lines
    ]


def _is_parsable(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def get_prompt_code(source_code: str, comment_data: dict) -> str:
    PLACEHOLDER_COMMENT = "Add the comment here"

    source_lines = source_lines_of(source_code)

    target_start = comment_data["start_line"]
    target_end = comment_data["end_line"]

    start, end = _scope_bounds(source_code, source_lines, target_start, target_end)

    output_lines: list[str] = []
    placeholder_placed = False
    for line_no in range(start, end + 1):
        line = source_lines[line_no - 1]
        line_is_target = target_start <= line_no <= target_end
        if not line_is_target:
            output_lines.append(line)
            continue
        if comment_data["type"] == "inline":
            # Keep the code the inline comment sat on, swapping the comment
            # for the placeholder.
            anchor = comment_data.get("anchor")
            code = anchor if anchor is not None else line.split("#", 1)[0].rstrip()
            if code:
                output_lines.append(f"{code}  # {PLACEHOLDER_COMMENT}")
                placeholder_placed = True
        elif line_no == target_start:
            # Collapse the block target to one placeholder line, keeping the
            # original indentation.
            indent = line[: len(line) - len(line.lstrip())]
            output_lines.append(f"{indent}# {PLACEHOLDER_COMMENT}")
            placeholder_placed = True

    if not placeholder_placed:
        raise ValueError(
            f"could not place the comment placeholder for the {comment_data['type']} "
            f"comment at line {target_start}: scope spans lines {start}-{end}"
        )

    # Check the dedented text, since that is what callers get.
    prompt_code = "\n".join(_strip_enclosing_indent(output_lines))
    if not _is_parsable(prompt_code):
        raise ValueError(
            f"the code block for the {comment_data['type']} comment at line "
            f"{target_start} is not parsable: scope spans lines {start}-{end}"
        )

    return prompt_code
