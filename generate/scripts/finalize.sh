#!/bin/bash
#SBATCH --job-name=generate-finalize
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/generate-finalize-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1

python -m generate.finalize \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}"
