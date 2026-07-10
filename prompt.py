from pathlib import Path

STATUS_TEMPLATE_PATHS = {
    "added": Path(__file__).parent / "prompts" / "add_comment.md",
    "modified": Path(__file__).parent / "prompts" / "modify_comment.md",
}

INTENT_INSTRUCTION_PATHS = {
    "what": Path(__file__).parent / "prompts" / "what.md",
    "how": Path(__file__).parent / "prompts" / "how.md",
    "why": Path(__file__).parent / "prompts" / "why.md",
}


def _code_type_instruction(comment_data: dict) -> str:
    if comment_data["type"] == "inline":
        return "Output a single comment line beginning with `#`"
    return "Output one or more comment lines, each beginning with `#`"


def _intent_instruction(intent: str) -> str:
    return INTENT_INSTRUCTION_PATHS[intent].read_text(encoding="utf-8")


def _build_modify_prompt(
    repo_name: str,
    filepath: str,
    comment_data: dict,
    diff_hunk: str,
    unmodified_comment: str,
) -> str:
    template = STATUS_TEMPLATE_PATHS["modified"].read_text(encoding="utf-8")
    return (
        template.replace("{repo_name}", repo_name)
        .replace("{filepath}", filepath)
        .replace("{comment_type}", comment_data["type"])
        .replace("{diff_hunk}", diff_hunk)
        .replace(
            "{unmodified_comment}",
            unmodified_comment,
        )
    )


def _build_add_prompt(
    repo_name: str, filepath: str, comment_data: dict, intent: str, scope_code: str
) -> str:
    template = STATUS_TEMPLATE_PATHS["added"].read_text(encoding="utf-8")
    return (
        template.replace("{repo_name}", repo_name)
        .replace("{filepath}", filepath)
        .replace("{comment_type}", comment_data["type"])
        .replace("{scope_code}", scope_code)
        .replace("{intent_instruction}", _intent_instruction(intent))
        .replace("{code_type_instruction}", _code_type_instruction(comment_data))
    )


def build_prompt(
    repo_name: str,
    filepath: str,
    comment_data: dict,
    status: str,
    intent: str,
    scope_code: str | None = None,
    diff_hunk: str | None = None,
    unmodified_comment: str | None = None,
) -> str:
    if status == "modified":
        if diff_hunk is None:
            raise ValueError("Diff hunk is required for modified comments")
        if unmodified_comment is None:
            raise ValueError("Unmodified comment is required for modified comments")
        return _build_modify_prompt(
            repo_name, filepath, comment_data, diff_hunk, unmodified_comment
        )

    if scope_code is None:
        raise ValueError("Scope code is required for added comments")

    return _build_add_prompt(repo_name, filepath, comment_data, intent, scope_code)
