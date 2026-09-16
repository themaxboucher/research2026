from functools import lru_cache

from transformers import GenerationConfig, pipeline

# Each job array task generates with exactly one model, so caching a second one
# would only compete for GPU memory with the one actually in use.
MAX_CACHED_MODELS = 1
MAX_OUTPUT_TOKENS = 512


@lru_cache(maxsize=MAX_CACHED_MODELS)
def _load_text_generation_pipeline(model_name: str):
    return pipeline("text-generation", model=model_name, device_map="auto")


def _raise_if_exceeds_context(text_generation_pipeline, messages: list[dict]) -> None:
    # The tokenizer's model_max_length is unreliable (a sentinel for CodeLlama,
    # the unenabled YaRN length for Qwen2.5), so trust the model config instead.
    context_limit = text_generation_pipeline.model.config.max_position_embeddings
    # Count the prompt the way the pipeline tokenizes it, chat template included
    prompt_tokens = len(
        text_generation_pipeline.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )["input_ids"]
    )
    if prompt_tokens + MAX_OUTPUT_TOKENS > context_limit:
        raise ValueError(
            f"Prompt has {prompt_tokens} tokens; with {MAX_OUTPUT_TOKENS} output "
            f"tokens it exceeds the {context_limit}-token context limit"
        )


def get_completion(model_name: str, prompt: str) -> str:
    text_generation_pipeline = _load_text_generation_pipeline(model_name)
    messages = [{"role": "user", "content": prompt}]
    _raise_if_exceeds_context(text_generation_pipeline, messages)
    output = text_generation_pipeline(
        messages,
        generation_config=GenerationConfig(max_new_tokens=MAX_OUTPUT_TOKENS),
    )
    return output[0]["generated_text"][-1]["content"]
