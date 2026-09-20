#!/bin/bash
#SBATCH --job-name=context-audit
#SBATCH --partition=cpu2019,cpu2021,cpu2022,cpu2019-bf05
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=logs/context-audit-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

source .venv/bin/activate

export TQDM_DISABLE=1

python -m generate.context_audit
