import textwrap

import pytest

from edits import SearchReplaceEdit, apply_edits, parse_edit_response


SIMPLE_FILE = textwrap.dedent("""\
    import os

    def read_config(path):
        # old comment about parsing
        value = os.environ.get(path)
        return value
""")


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


# parse_edit_response


def test_parse_edit_response_no_edits_returns_empty_list():
    assert parse_edit_response("NO_EDITS") == []


def test_parse_edit_response_no_edits_with_surrounding_whitespace():
    assert parse_edit_response("  NO_EDITS\n") == []


def test_parse_edit_response_extracts_single_block():
    response = _block("x = 1  # one", "x = 1  # uno")
    assert parse_edit_response(response) == [
        SearchReplaceEdit("x = 1  # one", "x = 1  # uno")
    ]


def test_parse_edit_response_extracts_multiple_blocks():
    response = _block("x = 1", "x = 1  # a") + "\n" + _block("y = 2", "y = 2  # b")
    assert parse_edit_response(response) == [
        SearchReplaceEdit("x = 1", "x = 1  # a"),
        SearchReplaceEdit("y = 2", "y = 2  # b"),
    ]


def test_parse_edit_response_ignores_filename_and_fences_around_block():
    response = "src/app.py\n```python\n" + _block("x = 1", "x = 1  # a") + "\n```"
    assert parse_edit_response(response) == [SearchReplaceEdit("x = 1", "x = 1  # a")]


def test_parse_edit_response_keeps_empty_replace_for_deletion():
    response = _block("# stale comment", "")
    assert parse_edit_response(response) == [SearchReplaceEdit("# stale comment", "")]


def test_parse_edit_response_raises_without_blocks_or_marker():
    with pytest.raises(ValueError, match="neither"):
        parse_edit_response("Here are my thoughts about the file...")


def test_parse_edit_response_raises_on_block_missing_divider():
    response = "<<<<<<< SEARCH\nx = 1\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="Malformed"):
        parse_edit_response(response)


# apply_edits


def test_apply_edits_with_no_edits_returns_file_unchanged():
    assert apply_edits(SIMPLE_FILE, []) == SIMPLE_FILE


def test_apply_edits_replaces_block_comment():
    edit = SearchReplaceEdit(
        "    # old comment about parsing",
        "    # reads the value from the environment",
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    assert "# reads the value from the environment" in result
    assert "# old comment about parsing" not in result


def test_apply_edits_removes_comment():
    edit = SearchReplaceEdit(
        "    # old comment about parsing\n    value = os.environ.get(path)",
        "    value = os.environ.get(path)",
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    assert "# old comment about parsing" not in result
    assert "value = os.environ.get(path)" in result


def test_apply_edits_adds_inline_comment():
    edit = SearchReplaceEdit(
        "    value = os.environ.get(path)",
        "    value = os.environ.get(path)  # may be None",
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    assert "value = os.environ.get(path)  # may be None" in result


def test_apply_edits_adds_docstring_after_def_line():
    edit = SearchReplaceEdit(
        "def read_config(path):",
        'def read_config(path):\n    """Read a config value from the environment."""',
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    assert '"""Read a config value from the environment."""' in result
    assert "# old comment about parsing" in result


def test_apply_edits_adds_module_docstring_at_file_start():
    edit = SearchReplaceEdit("import os", '"""Config helpers."""\nimport os')
    result = apply_edits(SIMPLE_FILE, [edit])
    assert result.startswith('"""Config helpers."""\nimport os\n')


def test_apply_edits_preserves_code_lines_exactly():
    edit = SearchReplaceEdit(
        "    # old comment about parsing\n    value = os.environ.get(path)",
        "    value = os.environ.get(path)",
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    code_lines = [line for line in result.splitlines() if line.strip()]
    expected_code_lines = [
        "import os",
        "def read_config(path):",
        "    value = os.environ.get(path)",
        "    return value",
    ]
    assert code_lines == expected_code_lines


def test_apply_edits_applies_multiple_edits_in_order():
    edits = [
        SearchReplaceEdit("import os", "import os  # standard library"),
        SearchReplaceEdit("    return value", "    return value  # may be None"),
    ]
    result = apply_edits(SIMPLE_FILE, edits)
    assert "import os  # standard library" in result
    assert "    return value  # may be None" in result


def test_apply_edits_matches_despite_trailing_whitespace_in_search():
    edit = SearchReplaceEdit(
        "    # old comment about parsing   ",
        "    # reads the value from the environment",
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    assert "# reads the value from the environment" in result
    assert "# old comment about parsing" not in result


def test_apply_edits_reindents_replace_when_search_is_dedented():
    edit = SearchReplaceEdit(
        "# old comment about parsing\nvalue = os.environ.get(path)",
        "# reads the value from the environment\nvalue = os.environ.get(path)",
    )
    result = apply_edits(SIMPLE_FILE, [edit])
    assert "    # reads the value from the environment" in result
    assert "    value = os.environ.get(path)" in result


def test_apply_edits_raises_when_search_matches_nothing():
    edit = SearchReplaceEdit("def totally_different():", "def totally_different():")
    with pytest.raises(ValueError, match="does not match"):
        apply_edits(SIMPLE_FILE, [edit])


def test_apply_edits_raises_when_search_is_ambiguous():
    file_content = "x = 1\ny = 2\nx = 1\ny = 2\n"
    edit = SearchReplaceEdit("x = 1", "x = 1  # first")
    with pytest.raises(ValueError, match="multiple locations"):
        apply_edits(file_content, [edit])


def test_apply_edits_raises_when_search_is_empty():
    with pytest.raises(ValueError, match="empty"):
        apply_edits(SIMPLE_FILE, [SearchReplaceEdit("", "# new comment")])
