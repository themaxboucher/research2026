#!/bin/bash
#SBATCH --job-name=prune-run
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/prune-run-%j.out

# Prunes a run's generated.jsonl and generated_scored.jsonl in place down to the
# comments still in its dataset, so back them up first. Rerun eval.finalize
# --skip-merge afterwards to recompute the metrics.
#
# Usage: sbatch [--export=ALL,DATASET_DIR=<timestamp>,RUN_DIR=<timestamp>] generate/scripts/prune.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1

python -m generate.prune \
  ${DATASET_DIR:+--dataset-dir "${DATASET_DIR}"} \
  ${RUN_DIR:+--run-dir "${RUN_DIR}"}
