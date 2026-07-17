#!/bin/bash
#SBATCH --job-name=mine-finalize
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/mine-finalize-%j.out
#SBATCH --error=logs/mine-finalize-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1

python -m collect.collect --finalize --run-dir "${RUN_DIR}"
