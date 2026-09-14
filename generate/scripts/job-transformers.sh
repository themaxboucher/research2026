#!/bin/bash
#SBATCH --job-name=generate-comments
# Every GPU in these partitions (L40 48 GB, A100/H100 80 GB, H200 141 GB) fits an
# 8B model in bf16; gpu-v100's 16 GB cards don't
#SBATCH --partition=gpu-h100,gpu-a100,gpu-l40
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/generate-comments-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

# Fail fast rather than silently loading the model on CPU when the GPU stack is
# broken, e.g. a driver too old for torch's CUDA build
python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is unavailable: check nvidia-smi and the torch CUDA build")'

export TQDM_DISABLE=1
export MODEL_PROFILE=transformers
# Models were pre-downloaded by submit.sh. Compute nodes can stay offline
export HF_HUB_OFFLINE=1

python -m generate.generate \
  --dataset-dir "${DATASET_DIR}" \
  --run-dir "${RUN_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"
