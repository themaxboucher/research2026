#!/bin/bash
#SBATCH --job-name=mine-finalize
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/mine-finalize-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1

python -m collect.finalize --dataset-dir "${DATASET_DIR}"
