#!/bin/bash
#SBATCH --job-name=analyze
# 256G needs cpu2023 (512 GB), cpu2025 (1 TB) or the cpu2021-bf24 backfill nodes
# (381 GB); cpu2019/2021/2022 nodes cap below it
#SBATCH --partition=cpu2023,cpu2025,cpu2021-bf24
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --output=logs/analyze-%j.out

set -euo pipefail

# Submit from the repository root after running: mkdir -p logs
# Optional arguments are forwarded to analyze.analyze.
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

python -m analyze.analyze "$@"
