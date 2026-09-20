#!/bin/bash
#SBATCH --job-name=correlation
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --output=logs/correlation-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module load python/3.13
source .venv/bin/activate

python -m analyze.correlation "$@"
