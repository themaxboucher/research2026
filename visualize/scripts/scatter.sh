#!/bin/bash
#SBATCH --job-name=scatter
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/scatter-%j.out

set -euo pipefail

# Submit from the repository root after running: mkdir -p logs
# Run after analysis has added complexity metrics to the scored file.
# Optional arguments are forwarded to visualize.scatter.
cd "${SLURM_SUBMIT_DIR}"

module load python/3.13
source .venv/bin/activate

python -m visualize.scatter "$@"
