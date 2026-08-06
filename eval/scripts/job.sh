#!/bin/bash
#SBATCH --job-name=evaluate-comments
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/evaluate-comments-%A_%a.out
#SBATCH --error=logs/evaluate-comments-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1

# Weights and metric scripts were pre-cached by submit.sh. The compute node can stay offline.
export HF_HUB_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m eval.eval \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"
