import pytest

from generate import (
    GenerationTask,
    _assert_code_preserved,
    _assert_valid_python_syntax,
    _build_comment_free_diff,
    _build_prompt,
)


# _assert_valid_python_syntax


def test_assert_valid_python_syntax_passes_for_valid_source():
    _assert_valid_python_syntax("x = 1\n")


def test_assert_valid_python_syntax_raises_value_error_for_invalid_syntax():
    with pytest.raises(ValueError, match="not valid Python"):
        _assert_valid_python_syntax("def foo(:\n")


# _assert_code_preserved


def test_assert_code_preserved_accepts_comment_only_differences():
    original = "x = 1  # old comment\ny = 2\n"
    generated = "x = 1  # new comment\ny = 2\n"
    _assert_code_preserved(generated, original)


def test_assert_code_preserved_raises_when_code_changes():
    original = "x = 1\ny = 2\n"
    generated = "x = 1\ny = 3\n"
    with pytest.raises(ValueError, match="altered the code"):
        _assert_code_preserved(generated, original)


# _build_comment_free_diff


def test_build_comment_free_diff_shows_code_change():
    diff = _build_comment_free_diff("x = 1\n", "x = 2\n", "src/app.py")
    assert "--- a/src/app.py" in diff
    assert "+++ b/src/app.py" in diff
    assert "-x = 1" in diff
    assert "+x = 2" in diff


def test_build_comment_free_diff_handles_missing_previous_code():
    diff = _build_comment_free_diff(None, "x = 1\n", "src/new.py")
    assert "+x = 1" in diff
    removed_lines = [
        line
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    assert removed_lines == []


# _build_prompt


def test_build_prompt_fills_all_placeholders():
    task = GenerationTask(
        filepath="src/app.py",
        code_with_outdated_comments="x = 1  # stale\n",
        comment_free_diff="+x = 1\n",
        previous_code=None,
    )
    prompt = _build_prompt(task)
    assert "{file_path}" not in prompt
    assert "{code_file}" not in prompt
    assert "{diff}" not in prompt
    assert 'path="src/app.py"' in prompt
    assert "x = 1  # stale" in prompt
    assert "+x = 1" in prompt
