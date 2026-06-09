# Research 2026

Compare how human developers write Python `#` comments with how LLMs write them on the same source files.

## Pipeline

The project runs in three stages. Each stage reads the previous data and writes the next under the run directory:

| Stage    | Command flag | Input                   | Output                  |
| -------- | ------------ | ----------------------- | ----------------------- |
| Collect  | `--collect`  | —                       | `files.jsonl`           |
| Generate | `--generate` | `files.jsonl`           | `files_generated.jsonl` |
| Report   | `--report`   | `files_generated.jsonl` | `reports/`              |

Run stages in that order: **collect → generate → report**.

1. **Collect** — Uses the GitHub REST API to fetch newly added `.py` files from high-star public repos after a cutoff date. Parses comments and docstrings and saves stripped source plus comment metadata.
2. **Generate** — Prompts one or more LLMs to add comments to the stripped files, saving each model's generated source and comments alongside the original. The goal is to compare multiple models and prompting strategies over time.
3. **Report** — Computes comment metrics directly from `files_generated.jsonl` and writes `summary.csv` (one row per repo × source, with total/inline/block/docstring comment counts and density statistics) plus histograms comparing the original human comments against every LLM.

## Running on an HPC cluster

`job.sh` is a SLURM batch script. Submit it on the cluster with `sbatch job.sh`. It writes results to `runs/<timestamp>/` and SLURM logs to `logs/`.
