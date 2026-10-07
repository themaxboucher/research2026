#!/bin/bash
#SBATCH --job-name=evaluate-comments
# 12 hours rules out ARC's 5-hour backfill partitions but fits the 24-hour ones
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2021-bf24,cpu2022-bf24
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/evaluate-comments-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1

# Weights and metric scripts were pre-cached by submit.sh. The compute node can stay offline.
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m eval.eval \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  ${FORCE:+--force}
