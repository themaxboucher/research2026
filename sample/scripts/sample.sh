#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/sample-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module load python/3.13
source .venv/bin/activate

python -m sample.sample "$@"
