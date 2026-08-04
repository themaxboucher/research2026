def normalize_comment(text: str) -> str:
    """Comment text reduced to its content. Per-line '#' markers and indentation
    are stripped. Everything is joined into a single line with single spaces."""
    words = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        words.extend(stripped.split())
    return " ".join(words)