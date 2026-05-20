import io
import tokenize

def get_comments_from_file(file_content: str) -> list[dict]:
    if not file_content:
        return []

    try:
        tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []

    lines = file_content.splitlines(keepends=True)
    found_comments: list[tuple[int, str, bool]] = []

    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        row_idx = row - 1
        if row_idx < 0 or row_idx >= len(lines):
            continue
        line_text = lines[row_idx]
        prefix = line_text[:col]
        is_standalone = prefix.strip() == ""
        comment_text = tok.string.rstrip("\r\n")
        found_comments.append((row, comment_text, is_standalone))

    if not found_comments:
        return []

    found_comments.sort(key=lambda x: x[0])

    comment_blocks: list[dict] = []
    for line_no, text, is_standalone in found_comments:
        if (
            comment_blocks
            and is_standalone
            and comment_blocks[-1]["_standalone"]
            and comment_blocks[-1]["end_line"] + 1 == line_no
        ):
            comment_blocks[-1]["comment"] += "\n" + text
            comment_blocks[-1]["end_line"] = line_no
        else:
            comment_blocks.append({
                "comment": text,
                "start_line": line_no,
                "end_line": line_no,
                "_standalone": is_standalone,
            })

    for block in comment_blocks:
        block.pop("_standalone", None)

    return comment_blocks

def strip_comments_from_file(file_content: str) -> str:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))

    lines = file_content.splitlines(keepends=True)
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        line = lines[row - 1]
        lines[row - 1] = line[:col].rstrip() + "\n"

    return "".join(lines)
