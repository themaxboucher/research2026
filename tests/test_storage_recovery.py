import json

from collect import _clean_previous_data
from storage import drop_trailing_records, load_from_jsonl, truncate_broken_tail


def _write_lines(directory, filename, lines):
    path = directory / f"{filename}.jsonl"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _record_line(record):
    return json.dumps(record) + "\n"


# truncate_broken_tail


def test_truncate_missing_file_is_noop(tmp_path):
    assert truncate_broken_tail(tmp_path, "data") == 0


def test_truncate_empty_file_is_noop(tmp_path):
    _write_lines(tmp_path, "data", [])
    assert truncate_broken_tail(tmp_path, "data") == 0


def test_truncate_valid_file_is_noop(tmp_path):
    lines = [_record_line({"a": 1}), _record_line({"a": 2})]
    _write_lines(tmp_path, "data", lines)

    assert truncate_broken_tail(tmp_path, "data") == 0
    assert load_from_jsonl(tmp_path, "data") == [{"a": 1}, {"a": 2}]


def test_truncate_removes_partial_last_line(tmp_path):
    partial_line = '{"a": 2, "broken'
    _write_lines(tmp_path, "data", [_record_line({"a": 1}), partial_line])

    assert truncate_broken_tail(tmp_path, "data") == len(partial_line)
    assert load_from_jsonl(tmp_path, "data") == [{"a": 1}]


def test_truncate_removes_invalid_terminated_last_line(tmp_path):
    _write_lines(tmp_path, "data", [_record_line({"a": 1}), '{"broken": \n'])

    assert truncate_broken_tail(tmp_path, "data") > 0
    assert load_from_jsonl(tmp_path, "data") == [{"a": 1}]


def test_truncate_empties_file_containing_only_partial_line(tmp_path):
    path = _write_lines(tmp_path, "data", ['{"broken'])

    assert truncate_broken_tail(tmp_path, "data") == len('{"broken')
    assert path.read_bytes() == b""


def test_truncate_handles_lines_longer_than_scan_chunk(tmp_path):
    long_record = {"a": "x" * 50_000}
    _write_lines(
        tmp_path, "data", [_record_line(long_record), '{"broken']
    )

    assert truncate_broken_tail(tmp_path, "data") == len('{"broken')
    assert load_from_jsonl(tmp_path, "data") == [long_record]


# drop_trailing_records


def test_drop_missing_file_is_noop(tmp_path):
    assert drop_trailing_records(tmp_path, "data", lambda record: True) == 0


def test_drop_stops_at_first_kept_record(tmp_path):
    lines = [
        _record_line({"repo": "kept"}),
        _record_line({"repo": "dropped"}),
        _record_line({"repo": "dropped"}),
    ]
    _write_lines(tmp_path, "data", lines)

    removed = drop_trailing_records(
        tmp_path, "data", lambda record: record["repo"] == "dropped"
    )

    assert removed == 2
    assert load_from_jsonl(tmp_path, "data") == [{"repo": "kept"}]


def test_drop_keeps_everything_when_last_record_is_kept(tmp_path):
    lines = [_record_line({"repo": "dropped"}), _record_line({"repo": "kept"})]
    _write_lines(tmp_path, "data", lines)

    removed = drop_trailing_records(
        tmp_path, "data", lambda record: record["repo"] == "dropped"
    )

    assert removed == 0
    assert load_from_jsonl(tmp_path, "data") == [
        {"repo": "dropped"},
        {"repo": "kept"},
    ]


def test_drop_can_empty_the_file(tmp_path):
    path = _write_lines(tmp_path, "data", [_record_line({"repo": "dropped"})])

    assert drop_trailing_records(tmp_path, "data", lambda record: True) == 1
    assert path.read_bytes() == b""


# _clean_previous_data


def _data_record(repo_name, index):
    return {"repo_name": repo_name, "commit_hash": f"hash-{index}"}


def _mined_record(repo_name):
    return {"repo": repo_name, "error": None}


def test_clean_missing_files_is_noop(tmp_path):
    _clean_previous_data(tmp_path)


def test_clean_completed_run_is_noop(tmp_path):
    _write_lines(
        tmp_path,
        "repo_files",
        [_record_line(_data_record("owner/a", 1)), _record_line(_data_record("owner/b", 2))],
    )
    _write_lines(
        tmp_path,
        "mined_repos",
        [_record_line(_mined_record("owner/a")), _record_line(_mined_record("owner/b"))],
    )

    _clean_previous_data(tmp_path)

    assert len(load_from_jsonl(tmp_path, "repo_files")) == 2
    assert len(load_from_jsonl(tmp_path, "mined_repos")) == 2


def test_clean_removes_partial_data_line_of_interrupted_repo(tmp_path):
    _write_lines(
        tmp_path,
        "repo_files",
        [
            _record_line(_data_record("owner/a", 1)),
            _record_line(_data_record("owner/b", 2)),
            '{"repo_name": "owner/b", "commit_ha',
        ],
    )
    _write_lines(tmp_path, "mined_repos", [_record_line(_mined_record("owner/a"))])

    _clean_previous_data(tmp_path)

    assert load_from_jsonl(tmp_path, "repo_files") == [_data_record("owner/a", 1)]


def test_clean_removes_unmarked_trailing_repo_block(tmp_path):
    _write_lines(
        tmp_path,
        "repo_files",
        [
            _record_line(_data_record("owner/a", 1)),
            _record_line(_data_record("owner/b", 2)),
            _record_line(_data_record("owner/b", 3)),
        ],
    )
    _write_lines(tmp_path, "mined_repos", [_record_line(_mined_record("owner/a"))])

    _clean_previous_data(tmp_path)

    assert load_from_jsonl(tmp_path, "repo_files") == [_data_record("owner/a", 1)]


def test_clean_treats_repo_with_truncated_marker_as_unmined(tmp_path):
    _write_lines(
        tmp_path,
        "repo_files",
        [
            _record_line(_data_record("owner/a", 1)),
            _record_line(_data_record("owner/b", 2)),
        ],
    )
    _write_lines(
        tmp_path,
        "mined_repos",
        [_record_line(_mined_record("owner/a")), '{"repo": "owner/b", "er'],
    )

    _clean_previous_data(tmp_path)

    assert load_from_jsonl(tmp_path, "repo_files") == [_data_record("owner/a", 1)]
    assert load_from_jsonl(tmp_path, "mined_repos") == [_mined_record("owner/a")]


def test_clean_empties_data_when_no_repo_was_marked(tmp_path):
    _write_lines(
        tmp_path, "repo_files", [_record_line(_data_record("owner/a", 1))]
    )

    _clean_previous_data(tmp_path)

    assert load_from_jsonl(tmp_path, "repo_files") == []
