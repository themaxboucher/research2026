#!/bin/bash
#SBATCH --job-name=filter-dataset
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/filter-dataset-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

# Submitted as an array by collect/scripts/filter-submit.sh, which exports the
# array width the tasks partition the repos on.
: "${NUM_TASKS:?NUM_TASKS must be exported; submit with collect/scripts/filter-submit.sh}"

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1

python -m collect.filter \
  ${DATASET_DIR:+--dataset-dir "${DATASET_DIR}"} \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}"
