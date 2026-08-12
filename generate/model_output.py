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
