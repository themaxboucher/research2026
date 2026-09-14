#!/bin/bash
#SBATCH --job-name=generate-comments-openrouter
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/generate-comments-openrouter-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1
export MODEL_PROFILE=openrouter
# Inference runs through the OpenRouter API rather than a local GPU, so this
# task needs no GPU or model weights. The compute node must have outbound
# internet access, and OPENROUTER_API_KEY is loaded from .env by the provider.

python -m generate.generate \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"
