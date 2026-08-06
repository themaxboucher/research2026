#!/bin/bash
#
# Usage:
#   ./submit.sh                             Score the latest run of the latest dataset
#   ./submit.sh --dataset-dir <timestamp>   Evaluate a specific dataset (default: latest)
#   ./submit.sh --run-dir <timestamp>       Evaluate a specific run (default: latest in dataset)
#   ./submit.sh --num-tasks 16              Width of the scoring job array
#   ./submit.sh --array 3,7                 Submit only these task indices (resume)
#   ./submit.sh --skip-setup                Reuse the existing .venv; skip pip install

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

DATASET_DIR=""
RUN_DIR=""
NUM_TASKS=""
ARRAY_INDICES=""
SKIP_SETUP=""

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

# Set up the Python environment. --skip-setup reuses an existing .venv as-is
module load gcc arrow/25.0.0
module load python/3.13

if [[ -n "$SKIP_SETUP" ]]; then
  if [[ ! -d .venv ]]; then
    echo "No .venv found; run without --skip-setup first to create it." >&2
    exit 1
  fi
  source .venv/bin/activate
else
  if [[ ! -d .venv ]]; then
    echo "Creating virtual environment in .venv"
    python -m venv .venv
  fi
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

# Warm the shared HF cache from the login node so the offline compute node can score.
python - <<'EOF'
from huggingface_hub import snapshot_download
import evaluate

from eval.constants import BERTSCORE_MODEL

print(f"Ensuring {BERTSCORE_MODEL} is in the HF cache")
snapshot_download(BERTSCORE_MODEL)

for metric in ("rouge", "bleu", "bertscore"):
    print(f"Caching {metric} metric script")
    evaluate.load(metric)
EOF

# Phase 1: Prepare. Records the array width in the run's eval manifest so the
# tasks agree on the partitioning and a single failed shard can be resubmitted.
PREP_ARGS=()
[[ -n "$DATASET_DIR" ]] && PREP_ARGS+=(--dataset-dir "$DATASET_DIR")
[[ -n "$RUN_DIR" ]] && PREP_ARGS+=(--run-dir "$RUN_DIR")
[[ -n "$NUM_TASKS" ]] && PREP_ARGS+=(--num-tasks "$NUM_TASKS")

echo "+ python -m eval.prepare ${PREP_ARGS[*]}"
PREP_OUT="$(python -m eval.prepare "${PREP_ARGS[@]}")"

DATASET_DIR="$(grep '^DATASET_DIR=' <<<"$PREP_OUT" | cut -d= -f2-)"
RUN_DIR="$(grep '^RUN_DIR=' <<<"$PREP_OUT" | cut -d= -f2-)"
NUM_TASKS="$(grep '^NUM_TASKS=' <<<"$PREP_OUT" | cut -d= -f2-)"

if [[ -z "$DATASET_DIR" || -z "$RUN_DIR" || -z "$NUM_TASKS" ]]; then
  echo "Prep did not return DATASET_DIR/RUN_DIR/NUM_TASKS:" >&2
  echo "$PREP_OUT" >&2
  exit 1
fi

echo "Dataset dir: $DATASET_DIR"
echo "Run dir:     $RUN_DIR"
echo "Num tasks:   $NUM_TASKS"

# Phase 2: Submit the scoring array
ARRAY_SPEC="${ARRAY_INDICES:-0-$((NUM_TASKS - 1))}"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR" \
  eval/scripts/job.sh)
echo "Submitted scoring array $ARRAY_JOB_ID (--array=$ARRAY_SPEC)"

# Phase 3: Merge shards and compute metrics once every task succeeds
FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR" \
  eval/scripts/finalize.sh)
echo "Submitted finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
