#!/bin/bash
#SBATCH --job-name=filter-context-overflow
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/filter-context-overflow-%j.out

# Drops the context overflows from a dataset filtered before collect.filter had
# that step. Rewrites dataset.jsonl and manifest.json in place, so back them up
# first. Then prune the dataset's runs with generate/scripts/prune.sh.
#
# Usage: sbatch [--export=ALL,DATASET_DIR=<timestamp>] collect/scripts/filter-context-overflow.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1
# Tokenizers and configs were pre-downloaded by generate/scripts/submit.sh. The
# compute node can stay offline.
export HF_HUB_OFFLINE=1

python -m collect.filter_context_overflow ${DATASET_DIR:+--dataset-dir "${DATASET_DIR}"}
