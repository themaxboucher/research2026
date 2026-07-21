#!/bin/bash
#SBATCH --job-name=evaluate-comments
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/evaluate-comments-%j.out
#SBATCH --error=logs/evaluate-comments-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

module load python/3.13

source .venv/bin/activate

export TQDM_DISABLE=1
# Weights and metric scripts were pre-cached by submit.sh. The compute node can
# stay offline.
export HF_HUB_OFFLINE=1
# BERTScore (roberta-large) runs on CPU here, so cap torch's thread pool to the
# cores SLURM actually granted rather than every core on the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# dataset-dir/run-dir/force are optional; forward only the ones submit.sh set.
ARGS=()
[[ -n "${DATASET_DIR:-}" ]] && ARGS+=(--dataset-dir "${DATASET_DIR}")
[[ -n "${RUN_DIR:-}" ]] && ARGS+=(--run-dir "${RUN_DIR}")
[[ -n "${FORCE:-}" ]] && ARGS+=(--force)

# ${ARGS[@]+...} keeps the default no-arg run working under `set -u` even on
# older bash, where a bare "${ARGS[@]}" on an empty array is an unbound error.
python -m eval.eval ${ARGS[@]+"${ARGS[@]}"}
