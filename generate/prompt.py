from pathlib import Path

LOCATION_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "location_prompt.md"

REGENERATE_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "regenerate_prompt.md"


def _code_type_instruction(comment_data: dict) -> str:
    if comment_data["type"] == "inline":
        return "Output a single comment line beginning with `#`"
    return "Output one or more comment lines, each beginning with `#`"


def build_regenerate_prompt(
    repo_name: str,
    filepath: str,
    commit_message: str,
    scope_code: str,
) -> str:
    template = REGENERATE_TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{repo_name}", repo_name)
        .replace("{filepath}", filepath)
        .replace("{commit_message}", commit_message)
        .replace("{scope_code}", scope_code)
    )


def build_location_prompt(
    repo_name: str,
    filepath: str,
    comment_data: dict,
    intent: str | None,
    commit_message: str,
    scope_code: str | None = None,
) -> str:
    if scope_code is None:
        raise ValueError("Scope code is required")

    template = LOCATION_TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{repo_name}", repo_name)
        .replace("{filepath}", filepath)
        .replace("{comment_type}", comment_data["type"])
        .replace("{commit_message}", commit_message)
        .replace("{scope_code}", scope_code)
        .replace("{code_type_instruction}", _code_type_instruction(comment_data))
    )
