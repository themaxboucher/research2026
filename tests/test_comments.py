import textwrap

from comments import get_comments_from_file, strip_comments_from_file


MODULE_WITH_ALL_TYPES = textwrap.dedent("""\
    \"\"\"Module docstring.\"\"\"

    x = 1  # inline comment

    # block line one
    # block line two

    def foo():
        \"\"\"Function docstring.\"\"\"
        return x
""")


# get_comments_from_file


def test_get_comments_empty_string_returns_empty_list():
    assert get_comments_from_file("") == []


def test_get_comments_no_comments_returns_empty_list():
    assert get_comments_from_file("x = 1\ny = 2\n") == []


def test_get_comments_inline_type_and_position():
    source = "x = 1  # inline\n"
    results = get_comments_from_file(source, include_docstrings=False)
    assert len(results) == 1
    assert results[0]["type"] == "inline"
    assert results[0]["comment"] == "# inline"
    assert results[0]["start_line"] == results[0]["end_line"] == 1


def test_get_comments_consecutive_block_comments_merge_into_one():
    source = "# line one\n# line two\n# line three\n"
    results = get_comments_from_file(source, include_docstrings=False)
    assert len(results) == 1
    assert results[0]["type"] == "block"
    assert results[0]["comment"] == "# line one\n# line two\n# line three"
    assert results[0]["start_line"] == 1
    assert results[0]["end_line"] == 3


def test_get_comments_non_consecutive_block_comments_are_separate():
    source = "# first\n\n# second\n"
    results = get_comments_from_file(source, include_docstrings=False)
    assert len(results) == 2
    assert all(r["type"] == "block" for r in results)


def test_get_comments_module_docstring_detected():
    source = '"""Module doc."""\nx = 1\n'
    results = get_comments_from_file(source)
    docstrings = [r for r in results if r["type"] == "docstring"]
    assert len(docstrings) == 1
    assert '"""Module doc."""' in docstrings[0]["comment"]


def test_get_comments_function_docstring_detected():
    source = textwrap.dedent("""\
        def foo():
            \"\"\"Does foo.\"\"\"
            pass
    """)
    results = get_comments_from_file(source)
    docstrings = [r for r in results if r["type"] == "docstring"]
    assert len(docstrings) == 1
    assert docstrings[0]["start_line"] == 2


def test_get_comments_class_docstring_detected():
    source = textwrap.dedent("""\
        class Bar:
            \"\"\"Bar class.\"\"\"
            pass
    """)
    results = get_comments_from_file(source)
    docstrings = [r for r in results if r["type"] == "docstring"]
    assert len(docstrings) == 1


def test_get_comments_multiline_docstring_correct_line_range():
    source = textwrap.dedent("""\
        def foo():
            \"\"\"
            Line one.
            Line two.
            \"\"\"
            pass
    """)
    results = get_comments_from_file(source)
    docstrings = [r for r in results if r["type"] == "docstring"]
    assert len(docstrings) == 1
    assert docstrings[0]["start_line"] == 2
    assert docstrings[0]["end_line"] == 5


def test_get_comments_include_docstrings_false_excludes_docstrings():
    source = '"""Module doc."""\n# regular\n'
    results = get_comments_from_file(source, include_docstrings=False)
    assert all(r["type"] != "docstring" for r in results)
    assert len(results) == 1


def test_get_comments_non_docstring_string_literal_not_included():
    source = textwrap.dedent("""\
        def foo():
            x = 1
            "this is not a docstring"
            return x
    """)
    results = get_comments_from_file(source)
    assert all(r["type"] != "docstring" for r in results)


def test_get_comments_results_sorted_by_start_line():
    results = get_comments_from_file(MODULE_WITH_ALL_TYPES)
    start_lines = [r["start_line"] for r in results]
    assert start_lines == sorted(start_lines)


def test_get_comments_all_types_present_in_mixed_source():
    results = get_comments_from_file(MODULE_WITH_ALL_TYPES)
    types_found = {r["type"] for r in results}
    assert types_found == {"inline", "block", "docstring"}


# strip_comments_from_file


def test_strip_comments_empty_string_returns_empty_string():
    assert strip_comments_from_file("") == ""


def test_strip_comments_no_comments_unchanged():
    source = "x = 1\ny = 2\n"
    assert strip_comments_from_file(source) == source


def test_strip_comments_inline_comment_removed_code_preserved():
    source = "x = 1  # remove me\n"
    result = strip_comments_from_file(source, strip_docstrings=False)
    assert "# remove me" not in result
    assert "x = 1" in result


def test_strip_comments_block_comment_removed():
    source = "# remove me\nx = 1\n"
    result = strip_comments_from_file(source, strip_docstrings=False)
    assert "# remove me" not in result
    assert "x = 1" in result


def test_strip_comments_docstring_removed_when_strip_docstrings_true():
    source = '"""Module doc."""\nx = 1\n'
    result = strip_comments_from_file(source, strip_docstrings=True)
    assert '"""Module doc."""' not in result
    assert "x = 1" in result


def test_strip_comments_docstring_preserved_when_strip_docstrings_false():
    source = '"""Module doc."""\nx = 1\n'
    result = strip_comments_from_file(source, strip_docstrings=False)
    assert '"""Module doc."""' in result


def test_strip_comments_multiline_docstring_all_lines_removed():
    source = textwrap.dedent("""\
        def foo():
            \"\"\"
            Line one.
            Line two.
            \"\"\"
            return 1
    """)
    result = strip_comments_from_file(source, strip_docstrings=True)
    assert "Line one" not in result
    assert "Line two" not in result
    assert "return 1" in result


def test_strip_comments_removes_both_comments_and_docstrings():
    result = strip_comments_from_file(MODULE_WITH_ALL_TYPES, strip_docstrings=True)
    assert "Module docstring" not in result
    assert "inline comment" not in result
    assert "block line one" not in result
    assert "Function docstring" not in result
    assert "x = 1" in result
