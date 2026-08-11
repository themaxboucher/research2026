RAW_DATASET_FILENAME = "dataset_raw"
DATASET_FILENAME = "dataset"
REPOS_CACHE_FILENAME = "repos_cache"
MINED_REPOS_FILENAME = "mined_repos"

# === LLM knowledge cutoffs ===
# GPT-5.6 Luna (https://developers.openai.com/api/docs/models/gpt-5.6-luna): Feb 16, 2026
# Llama 3.1 8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# Qwen 2.5 7b (https://huggingface.co/Qwen/Qwen2.5-7B): Unknown, but we know the model was released in Sep 2024
LLM_CUTOFF_DATE = "2026-02-17" # After GPT-5.6 Luna's knowledge cutoff, to avoid data leakage

COLLECTION_END_DATE = "2026-08-01"

REPO_LANGUAGE = "Python"

DEFAULT_MAX_REPOS = 1000
DEFAULT_REPOS_PER_TASK = 10