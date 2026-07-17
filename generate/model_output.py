import re

_CODE_FENCE_PATTERN = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)
_XML_WRAPPER_PATTERN = re.compile(
    r"\A<(?P<tag>[A-Za-z][\w-]*)(?:\s[^>]*)?>(?P<body>.*)</(?P=tag)>\Z", re.DOTALL
)


def strip_output_wrappers(text: str) -> str:
    """Unwrap a model response that arrives fenced in a markdown code block or
    an XML tag, leaving just the response body. Applied repeatedly so nested
    wrappers (e.g. a code fence inside a tag) all peel away."""
    previous = None
    while text != previous:
        previous = text
        text = text.strip()
        fence_match = _CODE_FENCE_PATTERN.match(text)
        if fence_match:
            text = fence_match.group("body")
            continue
        xml_match = _XML_WRAPPER_PATTERN.match(text)
        if xml_match:
            text = xml_match.group("body")
    return text.strip()


_XML_TAG_LINE_PATTERN = re.compile(r"<(?P<tag>[A-Za-z][\w-]*)(?:\s[^>]*)?>")


def _without_surrounding_blank_lines(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def strip_code_output_wrappers(text: str) -> str:
    """Unwrap a code response from a markdown fence or an XML tag. Unlike
    strip_output_wrappers, this works line-wise and never strips the leading
    whitespace of the first code line — an indented scope (e.g. a method)
    must keep its indentation to stay comparable with the original code."""
    previous = None
    while text != previous:
        previous = text
        lines = _without_surrounding_blank_lines(text.splitlines())
        if not lines:
            return ""

        is_fenced = (
            len(lines) >= 2
            and lines[0].lstrip().startswith("```")
            and lines[-1].lstrip().startswith("```")
        )
        if is_fenced:
            lines = lines[1:-1]

        opening_tag = _XML_TAG_LINE_PATTERN.fullmatch(lines[0].strip())
        is_tag_wrapped = (
            not is_fenced
            and len(lines) >= 2
            and opening_tag is not None
            and lines[-1].strip() == f"</{opening_tag.group('tag')}>"
        )
        if is_tag_wrapped:
            lines = lines[1:-1]

        text = "\n".join(lines)
    return text
