import io
import tokenize

from storage import load_from_json

def count_loc(file_content: str) -> int:
    return len(file_content.splitlines())

def count_comments(file_content: str) -> int:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    return len([tok for tok in tokens if tok.type == tokenize.COMMENT])

def avg_comment_character_length(file_content: str) -> float:
    tokens = list(tokenize.tokenize(io.BytesIO(file_content.encode("utf-8")).readline))
    if len([tok for tok in tokens if tok.type == tokenize.COMMENT]) == 0:
        return 0
    return sum([len(tok.string.rstrip("\r\n")) for tok in tokens if tok.type == tokenize.COMMENT]) / len([tok for tok in tokens if tok.type == tokenize.COMMENT])

def analyse_file(file: dict) -> dict:
    loc = count_loc(file["content"])
    comments = count_comments(file["content"])
    comments_per_loc = comments / loc

    return {
        "loc": loc,
        "comments": comments,
        "comments_per_loc": comments_per_loc,
    }

def analyse_dataset() -> list[dict]:
    dataset = load_from_json("files")

    num_repositories = len(set([file["repo_name"] for file in dataset]))
    num_files = len(dataset)
    avg_loc = sum([count_loc(file["content"]) for file in dataset]) / num_files
    avg_comments = sum([count_comments(file["content"]) for file in dataset]) / num_files
    avg_comments_per_loc = sum([count_comments(file["content"]) / count_loc(file["content"]) for file in dataset]) / num_files
    avg_comment_length = sum([avg_comment_character_length(file["content"]) for file in dataset]) / num_files
    percentage_of_files_with_comments = (sum([1 if count_comments(file["content"]) > 0 else 0 for file in dataset]) / num_files) * 100

    print("Number of repositories: ", num_repositories)
    print("Number of files: ", num_files)
    print("Average lines of code per file: ", avg_loc)
    print("Average comments per file: ", avg_comments)
    print("Average comments per line of code per file: ", avg_comments_per_loc)
    print("Average comment length in characters: ", avg_comment_length)
    print("Percentage of files with comment(s): ", percentage_of_files_with_comments)
    print("--------------------------------")