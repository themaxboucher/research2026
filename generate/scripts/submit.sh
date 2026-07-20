#!/bin/bash
#
# Usage:
#   ./generate-submit.sh --num-tasks 8                Fresh generation across 8 GPUs
#   ./generate-submit.sh --dataset-dir <timestamp>    Generate for a specific dataset (default: latest)
#   ./generate-submit.sh --run-dir <timestamp>        Resume an existing generation (reuses its config)
#   ./generate-submit.sh --array 3,7                  Submit only these task indices (resume)
#   ./generate-submit.sh --approaches location        Approaches to run (default: location,regenerate)
#   ./generate-submit.sh --max-generate 100           Cap files sent to the LLMs

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

NUM_TASKS=""
DATASET_DIR=""
RUN_DIR=""
ARRAY_INDICES=""
APPROACHES=""
MAX_GENERATE=""

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --approaches) APPROACHES="$2"; shift 2 ;;
    --max-generate) MAX_GENERATE="$2"; shift 2 ;;
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

export MODEL_PROFILE=cluster

# Warm the shared HF cache from the login node
python - <<'EOF'
from huggingface_hub import snapshot_download

from generate.models import MODEL_PROFILES

for model_name in MODEL_PROFILES["cluster"].model_names:
    print(f"Ensuring {model_name} is in the HF cache")
    snapshot_download(model_name)
EOF

# Phase 1: Prepare
PREP_ARGS=()
[[ -n "$DATASET_DIR" ]] && PREP_ARGS+=(--dataset-dir "$DATASET_DIR")
[[ -n "$RUN_DIR" ]] && PREP_ARGS+=(--run-dir "$RUN_DIR")
[[ -n "$APPROACHES" ]] && PREP_ARGS+=(--approaches "$APPROACHES")
[[ -n "$MAX_GENERATE" ]] && PREP_ARGS+=(--max-generate "$MAX_GENERATE")
[[ -n "$NUM_TASKS" ]] && PREP_ARGS+=(--num-tasks "$NUM_TASKS")

echo "+ python -m generate.prepare ${PREP_ARGS[*]}"
PREP_OUT="$(python -m generate.prepare "${PREP_ARGS[@]}")"

# Read the printed variables
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

# Phase 2: Submit the jobs array
ARRAY_SPEC="${ARRAY_INDICES:-0-$((NUM_TASKS - 1))}"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR" \
  generate/scripts/job.sh)
echo "Submitted array job $ARRAY_JOB_ID (--array=$ARRAY_SPEC)"

# Phase 3: Merge the per-task shards once every task succeeds
FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR" \
  generate/scripts/finalize.sh)
echo "Submitted finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
