RAW_DATASET_FILENAME = "dataset_raw"
DATASET_FILENAME = "dataset"
REPOS_CACHE_FILENAME = "repos_cache"
MINED_REPOS_FILENAME = "mined_repos"

# === LLM knowledge cutoffs ===
# Llama 3.1 8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# CodeLlama 7b (https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf): Unknown, but we know the HF repo was last updated on March 14, 2024
# Qwen 2.5 7b (https://huggingface.co/Qwen/Qwen2.5-7B): Unknown, but we know the HF repo was last updated on September 25, 2024
# Deepseek Coder 6.7b (https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct): Unknown, but we know the HF repo was last updated on February 2, 2024
LLM_CUTOFF_DATE = "2024-09-26" # After Qwen 2.5 7b's knowledge cutoff, to avoid data leakage

COLLECTION_END_DATE = "2026-08-01"

REPO_LANGUAGE = "Python"

DEFAULT_MAX_REPOS = 1000
DEFAULT_REPOS_PER_TASK = 1