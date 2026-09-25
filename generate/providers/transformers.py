from functools import lru_cache

from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
    pipeline,
)

# Each job array task generates with exactly one model, so caching a second one
# would only compete for GPU memory with the one actually in use.
MAX_CACHED_MODELS = 1
MAX_OUTPUT_TOKENS = 512

# These repos name LlamaTokenizerFast as their tokenizer class, which transformers
# 5 maps to its SentencePiece LlamaTokenizer. That replaces their byte-level BPE
# pre-tokenizer and decoder, dropping every space and newline, so their
# tokenizer.json has to be loaded as is.
VERBATIM_TOKENIZER_MODELS = {"deepseek-ai/deepseek-coder-6.7b-instruct"}


@lru_cache(maxsize=MAX_CACHED_MODELS)
def _load_text_generation_pipeline(model_name: str):
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=load_tokenizer(model_name),
        device_map="auto",
    )


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str):
    if model_name in VERBATIM_TOKENIZER_MODELS:
        return PreTrainedTokenizerFast.from_pretrained(model_name)
    # We only ever do chat completion. CodeLlama's tokenizer otherwise registers
    # <FILL_ME> as a special token with no embedding row, so a literal <FILL_ME>
    # in the prompt code indexes past the embedding table on the GPU.
    return AutoTokenizer.from_pretrained(model_name, fill_token=None)


@lru_cache(maxsize=None)
def model_context_limit(model_name: str) -> int:
    # The tokenizer's model_max_length is unreliable (a sentinel for CodeLlama,
    # the unenabled YaRN length for Qwen2.5), so trust the model config instead.
    return AutoConfig.from_pretrained(model_name).max_position_embeddings


def context_budget_tokens(context_limit: int) -> int:
    return context_limit - MAX_OUTPUT_TOKENS


def prompt_token_count(tokenizer, prompt: str) -> int:
    # Count the prompt the way the pipeline tokenizes it, chat template included
    messages = [{"role": "user", "content": prompt}]
    return len(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]
    )


def _raise_if_exceeds_context(text_generation_pipeline, prompt: str) -> None:
    context_limit = text_generation_pipeline.model.config.max_position_embeddings
    prompt_tokens = prompt_token_count(text_generation_pipeline.tokenizer, prompt)
    if prompt_tokens > context_budget_tokens(context_limit):
        raise ValueError(
            f"Prompt has {prompt_tokens} tokens; with {MAX_OUTPUT_TOKENS} output "
            f"tokens it exceeds the {context_limit}-token context limit"
        )


def get_completion(model_name: str, prompt: str) -> str:
    text_generation_pipeline = _load_text_generation_pipeline(model_name)
    _raise_if_exceeds_context(text_generation_pipeline, prompt)
    messages = [{"role": "user", "content": prompt}]
    output = text_generation_pipeline(
        messages,
        generation_config=GenerationConfig(max_new_tokens=MAX_OUTPUT_TOKENS),
    )
    return output[0]["generated_text"][-1]["content"]
