#!/bin/bash
#SBATCH --job-name=mine-code-comments
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/mine-code-comments-%A_%a.out
#SBATCH --error=logs/mine-code-comments-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1

python -m collect.collect \
  --dataset-dir "${DATASET_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}"
