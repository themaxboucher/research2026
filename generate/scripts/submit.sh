#!/bin/bash
#
# Usage:
#   ./submit.sh --num-tasks 8                Fresh generation across 8 tasks
#   ./submit.sh --profile openrouter         Inference backend: transformers (local
#                                            GPU) or openrouter (API) (default: transformers)
#   ./submit.sh --dataset-dir <timestamp>    Generate for a specific dataset (default: latest)
#   ./submit.sh --run-dir <timestamp>        Resume an existing generation (reuses its config)
#   ./submit.sh --array 3,7                  Submit only these task indices (resume)
#   ./submit.sh --approaches location        Approaches to run (default: location,regenerate)
#   ./submit.sh --max-generate 100           Cap files sent to the LLMs
#   ./submit.sh --skip-setup                 Reuse the existing .venv; skip pip install

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

NUM_TASKS=""
PROFILE="transformers"
DATASET_DIR=""
RUN_DIR=""
ARRAY_INDICES=""
APPROACHES=""
MAX_GENERATE=""
SKIP_SETUP=""

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --approaches) APPROACHES="$2"; shift 2 ;;
    --max-generate) MAX_GENERATE="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

# Each profile pairs an inference backend with its own job script
case "$PROFILE" in
  transformers|openrouter) JOB_SCRIPT="generate/scripts/job-${PROFILE}.sh" ;;
  *) echo "Unknown --profile: $PROFILE (expected transformers or openrouter)" >&2; usage 1 ;;
esac

# Set up the Python environment. --skip-setup reuses an existing .venv as-is
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

export MODEL_PROFILE="$PROFILE"

# Warm the shared HF cache from the login node so the offline compute nodes can
# load the weights. Only the transformers backend runs models locally; the
# openrouter backend calls an API and needs no local weights.
if [[ "$PROFILE" == "transformers" ]]; then
  python - <<'EOF'
from huggingface_hub import snapshot_download

from generate.providers.models import MODEL_PROFILES

for model_name in MODEL_PROFILES["transformers"].model_names:
    print(f"Ensuring {model_name} is in the HF cache")
    snapshot_download(model_name)
EOF
fi

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
  "$JOB_SCRIPT")
echo "Submitted array job $ARRAY_JOB_ID (--array=$ARRAY_SPEC, --profile=$PROFILE)"

# Phase 3: Merge the per-task shards once every task succeeds
FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR" \
  generate/scripts/finalize.sh)
echo "Submitted finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
