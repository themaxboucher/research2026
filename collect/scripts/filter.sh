#!/bin/bash
#SBATCH --job-name=filter-dataset
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/filter-dataset-%j.out
#SBATCH --error=logs/filter-dataset-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1

python -m collect.filter ${DATASET_DIR:+--dataset-dir "${DATASET_DIR}"}
