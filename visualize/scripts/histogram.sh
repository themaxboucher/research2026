#!/bin/bash
#SBATCH --job-name=histogram
#SBATCH --partition=cpu2022
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --output=logs/histogram-%j.out

set -euo pipefail

# Submit from the repository root after running: mkdir -p logs
# Optional arguments are forwarded to visualize.histogram.
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

# The current histogram implementation loads the entire scored file into RAM.
python -m visualize.histogram "$@"
