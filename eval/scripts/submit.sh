#!/bin/bash
#
# Usage:
#   ./submit.sh                             Score the latest run of the latest dataset
#   ./submit.sh --dataset-dir <timestamp>   Evaluate a specific dataset (default: latest)
#   ./submit.sh --run-dir <timestamp>       Evaluate a specific run (default: latest in dataset)
#   ./submit.sh --force                     Recompute scores that already exist

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

DATASET_DIR=""
RUN_DIR=""
FORCE=""

usage() {
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

# Set up the Python environment
module load python/3.13

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv"
  python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Warm the shared HF cache from the login node so the offline compute node can
# score. BERTScore pulls roberta-large's weights; rouge, bleu, and bertscore
# each need their evaluate metric script cached.
python - <<'EOF'
from huggingface_hub import snapshot_download
import evaluate

print("Ensuring roberta-large is in the HF cache")
snapshot_download("roberta-large")

for metric in ("rouge", "bleu", "bertscore"):
    print(f"Caching {metric} metric script")
    evaluate.load(metric)
EOF

# Submit the scoring job. dataset-dir/run-dir/force are optional; only the ones
# that were set get forwarded, and eval.eval falls back to the latest otherwise.
SBATCH_EXPORT="ALL"
[[ -n "$DATASET_DIR" ]] && SBATCH_EXPORT+=",DATASET_DIR=$DATASET_DIR"
[[ -n "$RUN_DIR" ]] && SBATCH_EXPORT+=",RUN_DIR=$RUN_DIR"
[[ -n "$FORCE" ]] && SBATCH_EXPORT+=",FORCE=$FORCE"

JOB_ID=$(sbatch --parsable --export="$SBATCH_EXPORT" eval/scripts/job.sh)
echo "Submitted evaluation job $JOB_ID"
