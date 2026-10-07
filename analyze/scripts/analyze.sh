#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --partition=cpu2022
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --output=logs/analyze-%j.out

set -euo pipefail

# Submit from the repository root after running: mkdir -p logs
# Optional arguments are forwarded to analyze.analyze.
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

python -m analyze.analyze "$@"
