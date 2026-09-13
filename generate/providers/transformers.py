from functools import lru_cache

from transformers import GenerationConfig, pipeline

# Each job array task generates with exactly one model, so caching a second one
# would only compete for GPU memory with the one actually in use.
MAX_CACHED_MODELS = 1
MAX_OUTPUT_TOKENS = 1024


@lru_cache(maxsize=MAX_CACHED_MODELS)
def _load_text_generation_pipeline(model_name: str):
    return pipeline("text-generation", model=model_name, device_map="auto")


def get_completion(model_name: str, prompt: str) -> str:
    text_generation_pipeline = _load_text_generation_pipeline(model_name)
    messages = [{"role": "user", "content": prompt}]
    output = text_generation_pipeline(
        messages,
        generation_config=GenerationConfig(max_new_tokens=MAX_OUTPUT_TOKENS),
    )
    return output[0]["generated_text"][-1]["content"]
