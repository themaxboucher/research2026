from functools import lru_cache

from transformers import GenerationConfig, pipeline

MAX_CACHED_MODELS = 4
MAX_OUTPUT_TOKENS = 32768

@lru_cache(maxsize=MAX_CACHED_MODELS)
def _load_text_generation_pipeline(model_name: str):
    return pipeline("text-generation", model=model_name, device_map="auto")


def get_completion(model_name: str, prompt: str, max_new_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    text_generation_pipeline = _load_text_generation_pipeline(model_name)
    messages = [{"role": "user", "content": prompt}]
    output = text_generation_pipeline(
        messages,
        generation_config=GenerationConfig(max_new_tokens=max_new_tokens),
    )
    return output[0]["generated_text"][-1]["content"]
