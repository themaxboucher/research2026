#!/bin/bash
#SBATCH --job-name=scatter
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/scatter-%j.out

set -euo pipefail

# Submit from the repository root after running: mkdir -p logs
# Run after analysis has added complexity metrics to the scored file.
# Optional arguments are forwarded to visualize.scatter.
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

python -m visualize.scatter "$@"
