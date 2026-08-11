import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from collect.filter_rules import target_comments
from generate.model_output import strip_output_wrappers
from generate.parse_code import scope_code
from generate.prompt import build_location_prompt
from generate.providers.models import ModelProfile


def _generate_with_llm(
    prompt: str,
    filepath: str,
    model_name: str,
    get_completion: Callable[[str, str], str],
) -> dict:
    raw_response = None
    try:
        raw_response = get_completion(model_name, prompt)
        comment_text = strip_output_wrappers(raw_response)
        if not comment_text:
            raise ValueError("Model returned an empty comment")
    except Exception as error:
        logging.warning(
            "Failed to generate comment in %s with model %s: %s",
            filepath,
            model_name,
            error,
        )
        return {
            "model": model_name,
            "raw_response": raw_response,
            "comment_text": None,
            "error": str(error),
        }
    return {
        "model": model_name,
        "raw_response": raw_response,
        "comment_text": comment_text,
        "error": None,
    }


def _run_llms_concurrently(
    prompt: str,
    filepath: str,
    model_profile: ModelProfile,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(model_profile.model_names)) as executor:
        futures = [
            executor.submit(
                _generate_with_llm,
                prompt,
                filepath,
                model_name,
                model_profile.get_completion,
            )
            for model_name in model_profile.model_names
        ]
        return [future.result() for future in as_completed(futures)]


def _comment_generation(
    file_data: dict,
    comment_data: dict,
    model_profile: ModelProfile,
) -> dict:
    source_code = file_data["source_code"]
    filepath = file_data["new_path"]
    intent = comment_data.get("intent")

    code = scope_code(source_code, comment_data)
    prompt = build_location_prompt(
        file_data["repo_name"],
        filepath,
        comment_data,
        commit_message=file_data["commit_message"],
        status="added",
        intent=intent,
        scope_code=code,
    )

    results = _run_llms_concurrently(prompt, filepath, model_profile)
    return {
        "type": comment_data["type"],
        "status": comment_data["status"],
        "start_line": comment_data["start_line"],
        "end_line": comment_data["end_line"],
        "anchor": comment_data.get("anchor"),
        "comment": comment_data.get("comment"),
        "prompt": prompt,
        "results": results,
    }


def location_generate_for_file(file_data: dict, model_profile: ModelProfile) -> dict:
    comment_generations = []
    for comment_data in target_comments(file_data):
        try:
            comment_generations.append(
                _comment_generation(file_data, comment_data, model_profile)
            )
        except Exception as error:
            logging.warning(
                "Skipping a comment in %s: could not build generation inputs: %s",
                file_data.get("new_path"),
                error,
            )

    return {
        "repo_name": file_data["repo_name"],
        "commit_hash": file_data.get("commit_hash"),
        "new_path": file_data["new_path"],
        "comment_generations": comment_generations,
    }
