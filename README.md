# Research 2026

Compare how human developers write Python `#` comments with how LLMs write them on the same source files.

## Pipeline

The project runs in three stages. Each stage reads the previous data and writes the next under `data/`:

| Stage | Command flag | Input | Output |
|-------|--------------|-------|--------|
| Collect | `--collect` | — | `data/files.json` |
| Generate | `--generate` | `files.json` | `data/files_generated.json` |
| Analyse | `--analyse` | `files_generated.json` | `data/files_analysed.json` |

Run stages in that order: **collect → generate → analyse**.

1. **Collect** — Uses the GitHub REST API to fetch newly added `.py` files from high-star public repos after a cutoff date. Parses `#` line comments (not docstrings) and saves stripped source plus comment metadata.
2. **Generate** — Prompts LLMs to add comments to the stripped files. The goal is to compare multiple models and prompting strategies over time. Right now this stage uses OpenRouter.
3. **Analyse** — Computes metrics and classifications on human and LLM comments so the two can be compared.
