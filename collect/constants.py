RAW_DATASET_FILENAME = "dataset_raw"
DATASET_FILENAME = "dataset"
REPOS_CACHE_FILENAME = "repos_cache"
MINED_REPOS_FILENAME = "mined_repos"

# === LLM knowledge cutoffs ===
# Llama 3.1 8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# CodeLlama 7b (https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf): Unknown, but we know the model was released on August 24, 2023 (https://about.fb.com/news/2023/08/code-llama-ai-for-coding/)
# Qwen 2.5 7b (https://huggingface.co/Qwen/Qwen2.5-7B): Unknown, but we know the model was released on September 18, 2024 (https://qwen.ai/blog?id=qwen2.5)
# Deepseek Coder 6.7b (https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct): Unknown, know the model was released in November 2023
LLM_CUTOFF_DATE = (
    "2024-09-19"  # After all LLM's knowledge cutoff, to avoid data leakage
)

COLLECTION_END_DATE = "2026-08-11"

REPO_LANGUAGE = "Python"

DEFAULT_MAX_REPOS = 1000
DEFAULT_REPOS_PER_TASK = 1
