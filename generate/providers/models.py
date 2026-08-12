import os
from typing import Callable, NamedTuple

from generate.providers import openrouter, transformers


class ModelProfile(NamedTuple):
    model_names: list[str]
    get_completion: Callable[[str, str], str]


MODEL_PROFILES = {
    "openrouter": ModelProfile(
        model_names=[
            "meta-llama/llama-3.1-8b-instruct",
            "qwen/qwen-2.5-7b-instruct",
        ],
        get_completion=openrouter.get_completion,
    ),
    "transformers": ModelProfile(
        model_names=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/CodeLlama-7b-Instruct-hf",
            "Qwen/Qwen2.5-7B-Instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ],
        get_completion=transformers.get_completion,
    ),
}
DEFAULT_MODEL_PROFILE = "transformers"


def get_model_profile() -> tuple[ModelProfile, str]:
    profile_name = os.environ.get("MODEL_PROFILE", DEFAULT_MODEL_PROFILE)
    if profile_name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown MODEL_PROFILE {profile_name!r}. "
            f"Expected one of: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[profile_name], profile_name
