#!/bin/bash
#SBATCH --job-name=mine-code-comments
# 256G needs cpu2023 (512 GB), cpu2025 (1 TB) or the cpu2021-bf24 backfill nodes
# (381 GB); cpu2019/2021/2022 nodes cap below it
#SBATCH --partition=cpu2023,cpu2025,cpu2021-bf24
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/mine-code-comments-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1

python -m collect.collect \
  --dataset-dir "${DATASET_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${NUM_TASKS}"
