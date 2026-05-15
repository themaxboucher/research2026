import io
import json
import os
import tokenize
from unidiff import PatchSet

from github import search_repos, get_repo_commits, get_commit, get_file_contents
from rich.console import Console
from rich.syntax import Syntax
from concurrent.futures import ThreadPoolExecutor

console = Console()

MAX_WORKERS = 20

CUTOFF_DATE = "2026-01-01"

# example_comment = {
#     "repo_owner": "Bob",
#     "repo_name": "cool-project",
#     "filepath": "src/main.py",
#     "comment": "# This is a comment",
#     "start_line": 1,
#     "end_line": 1,
# }

comments = []
files = []

def get_comments_from_patch(filepath: str, patch: str) -> list[dict]:
    # add header so the patch can be parsed by unidiff
    diff_text = f"--- a/{filepath}\n+++ b/{filepath}\n{patch}"
    
    # make sure the patch ends with a newline
    if not diff_text.endswith("\n"):
        diff_text += "\n"

    try:
        patch_set = PatchSet(diff_text)
    except Exception:
        raise ValueError(f"Failed to parse patch for {filepath}")

    if not patch_set:
        raise ValueError(f"No patch set found for {filepath}")

    patched_file = patch_set[0] # get the first and only patch set

    added_line_numbers: set[int] = set()
    for hunk in patched_file:
        for line in hunk:
            if line.is_added and line.target_line_no is not None:
                added_line_numbers.add(line.target_line_no)

    # return early if no lines were added
    if not added_line_numbers:
        return []

    found_comments: list[tuple[int, str, bool]] = []

    for hunk in patched_file:
        target_lines = [line for line in hunk if line.is_context or line.is_added]
        if not target_lines:
            continue

        hunk_start = target_lines[0].target_line_no
        hunk_source_lines = [line.value if line.value.endswith("\n") else line.value + "\n" for line in target_lines]
        hunk_source = "".join(hunk_source_lines)

        try:
            tokens = list(tokenize.tokenize(io.BytesIO(hunk_source.encode("utf-8")).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            continue

        for tok in tokens:
            if tok.type != tokenize.COMMENT:
                continue
            row, col = tok.start
            real_line_no = hunk_start + row - 1
            # skip if the comment is not on an added line
            if real_line_no not in added_line_numbers:
                continue
            row_idx = row - 1
            if row_idx < 0 or row_idx >= len(hunk_source_lines):
                continue
            line_text = hunk_source_lines[row_idx]
            prefix = line_text[:col]
            is_standalone = prefix.strip() == ""
            comment_text = tok.string.rstrip("\r\n")
            found_comments.append((real_line_no, comment_text, is_standalone))

    if not found_comments:
        return []

    found_comments.sort(key=lambda x: x[0]) # sort comments by line number

    comment_blocks: list[dict] = []
    for line_no, text, is_standalone in found_comments:
        # group multi line comments
        if (
            comment_blocks
            and is_standalone
            and comment_blocks[-1]["_standalone"]
            and comment_blocks[-1]["end_line"] + 1 == line_no
        ):
            comment_blocks[-1]["comment"] += "\n" + text
            comment_blocks[-1]["end_line"] = line_no
        else:
            comment_blocks.append({
                "filepath": filepath,
                "comment": text,
                "start_line": line_no,
                "end_line": line_no,
                "_standalone": is_standalone,
            })

    # remove the temporary _standalone key from the comment blocks
    for block in comment_blocks:
        block.pop("_standalone", None)

    return comment_blocks

def strip_comments_from_file(file_content: str) -> str:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))

    lines = file_content.splitlines(keepends=True)
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        line = lines[row - 1]
        lines[row - 1] = line[:col].rstrip() + "\n"

    return "".join(lines)

def save_to_json(data: list[dict], filename: str):
    os.makedirs("data", exist_ok=True)
    with open(f"data/{filename}.json", "w") as f:
        json.dump(data, f, indent=2)

def process_commit_file(repo: dict, commit: dict, file: dict) -> dict | None:
    if file.get("status") == "removed":
        return None
    filepath = file["filename"]
    is_python_file = filepath.endswith(".py")
    if not is_python_file:
        return None
    patch = file.get("patch")
    if not patch:
        return None
    commit_comments = get_comments_from_patch(filepath, patch)
    for comment in commit_comments:
        comment["repo_owner"] = repo["owner"]["login"]
        comment["repo_name"] = repo["name"]
    comments.extend(commit_comments)
    return get_file_contents(repo["owner"]["login"], repo["name"], filepath, commit["sha"])

def main():
    repo_search_results = search_repos(pushed_after=CUTOFF_DATE)

    print("Total repositories found:", len(repo_search_results))

    repos = repo_search_results[3:10] # only get some of the repos to save time

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        repo_commits = list[tuple[dict, list[dict]]](
            executor.map(
                lambda repo: (
                    repo,
                    get_repo_commits(repo["name"], repo["owner"]["login"], since=CUTOFF_DATE),
                ),
                repos,
            )
        )

        for repo, commits in repo_commits:
            print(f"{len(commits)} commits found in {repo['name']}")
            detailed_repo_commits = list[dict](executor.map(
                lambda commit: get_commit(repo["owner"]["login"], repo["name"], commit["sha"]),
                commits[:20], # only get the first 20 commits to save time
            ))
            for commit in detailed_repo_commits:
                files_contents = list[dict](executor.map(
                    lambda file: process_commit_file(repo, commit, file),
                    commit["files"],
                ))
                for file_content in files_contents:
                    if file_content:
                        file_info = {
                            "repo_owner": repo["owner"]["login"],
                            "repo_name": repo["name"],
                            "filepath": file_content["path"],
                            "sha": commit["sha"],
                            "content": file_content["content"],
                            "content_without_comments": strip_comments_from_file(file_content["content"]),
                        }
                        files.append(file_info)

if __name__ == "__main__":
    main()
    save_to_json(comments, "comments")
    save_to_json(files, "files")
    print("Total comments found:", len(comments))
    print("Total files found:", len(files))

    for file in files[:10]:
        print("File:", file["filepath"])
        print("Content:")
        console.print(Syntax(file["content"][:1000], "python", theme="monokai"))
        print("Content without comments:")
        console.print(Syntax(file["content_without_comments"][:1000], "python", theme="monokai"))
        print("-" * 80)
