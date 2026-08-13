import re

# Directive tokens that end in ":", which acts as the delimiter (e.g.
# "type: ignore", "type:ignore", "pragma: no cover"). Matched at the start of a
# comment body, case-insensitively, followed by a literal colon.
_COLON_DIRECTIVES = (
    "type",
    "pragma",
    "fmt",
    "isort",
    "pylint",
    "ruff",
    "mypy",
    "pyright",
    "doctest",
)

# Directive tokens that are bare words (e.g. "noqa", "noqa: BLE001", "nosec").
# Matched at the start of a comment body and must not run into a longer word, so
# "noqa" matches but "noqable" does not.
_WORD_DIRECTIVES = (
    "noqa",
    "nosec",
    "noinspection",
    # Dual-audience issue-tracker / IDE markers. Requested to count as
    # machine-directed so they are excluded from documentation edits.
    "todo",
    "fixme",
    "hack",
    "xxx",
)

_DIRECTIVE_RE = re.compile(
    r"^(?:"
    + "|".join(re.escape(word) for word in _WORD_DIRECTIVES)
    + r")(?![A-Za-z0-9])"
    r"|^(?:" + "|".join(re.escape(name) for name in _COLON_DIRECTIVES) + r"):",
    re.IGNORECASE,
)

# PEP 263 encoding declaration, e.g. "# -*- coding: utf-8 -*-" or "# coding: utf-8".
_ENCODING_RE = re.compile(r"coding[:=]\s*[-\w.]+", re.IGNORECASE)


def _line_is_directive(raw_line: str) -> bool:
    stripped = raw_line.strip()
    if not stripped.startswith("#"):
        return False
    # Shebang: "#!/usr/bin/env python" (no space between "#" and "!").
    if stripped.startswith("#!"):
        return True
    # Body after the leading "#" markers and surrounding whitespace.
    body = stripped.lstrip("#").strip()
    if not body:
        return False
    # Jupyter / VS Code cell marker: "# %%".
    if body.startswith("%%"):
        return True
    # PEP 263 encoding declaration.
    if _ENCODING_RE.search(body):
        return True
    return _DIRECTIVE_RE.match(body) is not None


def _is_machine_directive_comment(comment_text: str) -> bool:
    """Return True if *comment* is machine-directed rather than documentation.

    Machine-directed comments are consumed by tooling (linters, type checkers,
    formatters, coverage, doctest, encoding/interpreter, editors) rather than
    written for human readers. TODO/FIXME/HACK/XXX markers are treated as
    machine-directed too, since they are also parsed by issue trackers and IDEs.
    """
    return any(_line_is_directive(line) for line in comment_text.splitlines())


def _has_at_least_one_alpha_char(s: str) -> bool:
    """Check if a string contains at least one alphabetic character."""
    return any(char.isalpha() for char in s)


def _is_visual_separator_comment(comment: str) -> bool:
    """Check if a comment is a visual separator (e.g., a line of dashes or equals)."""
    # A separator is a run of at least 5 identical punctuation characters,
    # such as "-----", "=====", or "*****".
    run_length = 0
    previous_char = ""
    for char in comment:
        if char.isalnum() or char.isspace() or char == "#":
            run_length = 0
            previous_char = ""
            continue
        if char == previous_char:
            run_length += 1
        else:
            run_length = 1
            previous_char = char
        if run_length >= 5:
            return True
    return False


def is_ai_authored_file(source_file: dict) -> bool:
    AI_AUTHORED_IDENTIFIERS = {
        # Anthropic — Claude Code
        "Co-authored-by: Claude",
        "noreply@anthropic.com",
        "Generated with Claude Code",
        # GitHub Copilot — coding agent / Copilot CLI
        "Co-authored-by: Copilot",
        "Copilot@users.noreply.github.com",  # e.g. 198982749+Copilot@users.noreply.github.com
        "Copilot[bot]@users.noreply.github.com",
        # Cursor — background/cloud agent
        "Co-authored-by: Cursor",
        "cursoragent@cursor.com",
        "cursoragent@users.noreply.github.com",
        "Made-with: Cursor",
        # OpenAI — Codex CLI / Codex cloud
        "Co-authored-by: Codex",
        "noreply@openai.com",
        "chatgpt-codex-connector[bot]",
        # Google — Gemini CLI / Gemini Code Assist
        "Co-authored-by: Gemini",
        "gemini-code-assist[bot]",
        "gemini-cli@users.noreply.github.com",
        "gemini-cli-agent@google.com",
        # Aider
        "Co-authored-by: aider",
        "noreply@aider.chat",
        "aider@aider.chat",
        # Cognition — Devin
        "Co-authored-by: Devin",
        "devin-ai-integration",  # also covers the devin-ai-integration[bot] account
        # Generic / cross-tool markers
        "🤖 Generated with",
        "Assisted-by:",
        "Co-authored-by: AI",
    }
    if source_file.get("commit_message") is None:
        raise ValueError(
            "Commit message is required to determine if a file is AI-authored"
        )
    commit_message = source_file.get("commit_message").lower()
    for identifier in AI_AUTHORED_IDENTIFIERS:
        if identifier.lower() in commit_message:
            return True
    return False


def get_target_comments(comments: list[dict]) -> list[dict]:
    TARGET_COMMENT_TYPES = {"inline", "block"}
    TARGET_COMMENT_STATUSES = {"added"}
    # `None` means the comment's intent was never labeled; it gets the generic instruction
    TARGET_INTENTS = {"what", "why", "how", None}
    return [
        comment
        for comment in comments
        if comment.get("type") in TARGET_COMMENT_TYPES
        and comment.get("status") in TARGET_COMMENT_STATUSES
        and comment.get("comment") is not None
        and comment.get("intent") in TARGET_INTENTS
        and not _is_machine_directive_comment(comment["comment"])
        and _has_at_least_one_alpha_char(comment.get("comment", ""))
        and not _is_visual_separator_comment(comment.get("comment", ""))
        and comment.get("comment", "").isascii()  # Exclude non-english comments
    ]


def has_eligible_metadata(source_file: dict) -> bool:
    is_python_file = (source_file.get("new_path") or "").endswith(".py")
    if not is_python_file:
        return False

    ELIGIBLE_CHANGE_TYPES = {"MODIFY"}
    is_valid_change_type = source_file.get("change_type") in ELIGIBLE_CHANGE_TYPES
    if not is_valid_change_type:
        return False

    has_source_code = source_file.get("source_code") is not None
    if not has_source_code:
        return False

    has_previous_source_code = source_file.get("previous_source_code") is not None
    if not has_previous_source_code:
        return False

    has_commit_message = source_file.get("commit_message") is not None
    if not has_commit_message:
        return False

    return True
