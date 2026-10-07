#!/bin/bash
#SBATCH --job-name=correlation
#SBATCH --partition=cpu2022
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --output=logs/correlation-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

python -m analyze.correlation "$@"
