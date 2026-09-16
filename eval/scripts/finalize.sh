#!/bin/bash
#SBATCH --job-name=evaluate-finalize
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=logs/evaluate-finalize-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1
# Weights and metric scripts were pre-cached by submit.sh. The compute node can
# stay offline.
export HF_HUB_OFFLINE=1

python -m eval.finalize \
  --dataset-dir "${DATASET_DIR:-}" \
  --run-dir "${RUN_DIR:-}" \
  "$@"
