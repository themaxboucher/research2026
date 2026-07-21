#!/bin/bash
#SBATCH --job-name=generate-comments
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=logs/generate-comments-%A_%a.out
#SBATCH --error=logs/generate-comments-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1
export MODEL_PROFILE=transformers
# Models were pre-downloaded by submit.sh. Compute nodes can stay offline
export HF_HUB_OFFLINE=1

python -m generate.generate \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"
