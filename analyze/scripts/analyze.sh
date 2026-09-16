#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --output=logs/analyze-%j.out

set -euo pipefail

# Submit from the repository root after running: mkdir -p logs
# Optional arguments are forwarded to analyze.analyze.
cd "${SLURM_SUBMIT_DIR}"

module load python/3.13
source .venv/bin/activate

python -m analyze.analyze "$@"
