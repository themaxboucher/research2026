import pytest

from generate import _assert_valid_python_syntax, _parse_generated_code


# _assert_valid_python_syntax


def test_assert_valid_python_syntax_passes_for_valid_source():
    _assert_valid_python_syntax("x = 1\n")


def test_assert_valid_python_syntax_raises_value_error_for_invalid_syntax():
    with pytest.raises(ValueError, match="not valid Python"):
        _assert_valid_python_syntax("def foo(:\n")


# _parse_generated_code


def test_parse_generated_code_extracts_code_from_python_fence():
    response = "```python\nx = 1\n```"
    assert _parse_generated_code(response) == "x = 1\n"


def test_parse_generated_code_extracts_code_from_plain_fence():
    response = "```\nx = 1\n```"
    assert _parse_generated_code(response) == "x = 1\n"


def test_parse_generated_code_uses_full_response_when_no_fence():
    assert _parse_generated_code("x = 1\n") == "x = 1\n"


def test_parse_generated_code_appends_newline_when_missing():
    assert _parse_generated_code("x = 1").endswith("\n")


def test_parse_generated_code_does_not_double_newline():
    assert _parse_generated_code("x = 1\n") == "x = 1\n"


def test_parse_generated_code_raises_for_invalid_python_in_fence():
    with pytest.raises(ValueError, match="not valid Python"):
        _parse_generated_code("```python\ndef foo(:\n```")


def test_parse_generated_code_raises_for_invalid_python_without_fence():
    with pytest.raises(ValueError, match="not valid Python"):
        _parse_generated_code("def foo(:\n")
