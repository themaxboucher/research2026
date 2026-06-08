#!/bin/bash
#SBATCH --job-name=mine-code-comments
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/mine-code-comments-%j.out
#SBATCH --error=logs/mine-code-comments-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

if [[ ! -f .env ]]; then
  echo "Missing .env with GITHUB_TOKENS (or GITHUB_TOKEN)."
  exit 1
fi

module load python/3.13

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv"
  python -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export TQDM_DISABLE=1

python main.py --collect --smoke-test --new-run
