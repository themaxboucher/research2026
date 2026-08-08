#!/bin/bash
#SBATCH --job-name=evaluate-finalize
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/evaluate-finalize-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1
# Weights and metric scripts were pre-cached by submit.sh. The compute node can
# stay offline.
export HF_HUB_OFFLINE=1

python -m eval.finalize \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}"
