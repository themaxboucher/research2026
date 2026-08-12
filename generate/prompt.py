from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "prompts" / "location_prompt.md"


def _code_type_instruction(comment_data: dict) -> str:
    if comment_data["type"] == "inline":
        return "Output a single comment line beginning with `#`"
    return "Output one or more comment lines, each beginning with `#`"


def build_location_prompt(
    repo_name: str,
    filepath: str,
    comment_data: dict,
    commit_message: str,
    code: str,
) -> str:
    if code is None:
        raise ValueError("Code is required")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("{repo_name}", repo_name)
        .replace("{filepath}", filepath)
        .replace("{comment_type}", comment_data["type"])
        .replace("{commit_message}", commit_message)
        .replace("{code}", code)
        .replace("{code_type_instruction}", _code_type_instruction(comment_data))
    )
