#!/bin/bash
#
# Usage:
#   ./generate-submit.sh --num-tasks 8              Fresh generation across 8 GPUs
#   ./generate-submit.sh --run-dir runs/<ts> ...    Generate for a specific run (default: latest)
#   ./generate-submit.sh --generation <label>       Resume an existing generation (reuses its config)
#   ./generate-submit.sh --array 3,7                Submit only these task indices (resume)
#   ./generate-submit.sh --approaches location      Approaches to run (default: location,regenerate)
#   ./generate-submit.sh --max-generate 100         Cap files sent to the LLMs

set -euo pipefail

# Run everything from the repo root so .env, .venv, runs/ and logs/ resolve
cd "$(dirname "$0")/../.."

NUM_TASKS=""
RUN_DIR=""
GENERATION=""
ARRAY_INDICES=""
APPROACHES=""
MAX_GENERATE=""

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --generation) GENERATION="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --approaches) APPROACHES="$2"; shift 2 ;;
    --max-generate) MAX_GENERATE="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

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

from generate.generate import MODEL_PROFILES

for model_name in MODEL_PROFILES["cluster"].model_names:
    print(f"Ensuring {model_name} is in the HF cache")
    snapshot_download(model_name)
EOF

# Phase 1: Write the manifest the array tasks will run from
PREP_ARGS=(--prepare)
[[ -n "$RUN_DIR" ]] && PREP_ARGS+=(--run-dir "$RUN_DIR")
[[ -n "$GENERATION" ]] && PREP_ARGS+=(--generation "$GENERATION")
[[ -n "$NUM_TASKS" ]] && PREP_ARGS+=(--num-tasks "$NUM_TASKS")
[[ -n "$APPROACHES" ]] && PREP_ARGS+=(--approaches "$APPROACHES")
[[ -n "$MAX_GENERATE" ]] && PREP_ARGS+=(--max-generate "$MAX_GENERATE")

echo "+ python -m generate.generate ${PREP_ARGS[*]}"
PREP_OUT="$(python -m generate.generate "${PREP_ARGS[@]}")"

RUN_DIR="$(grep '^RUN_DIR=' <<<"$PREP_OUT" | cut -d= -f2-)"
GENERATION="$(grep '^GENERATION=' <<<"$PREP_OUT" | cut -d= -f2-)"
NUM_TASKS="$(grep '^NUM_TASKS=' <<<"$PREP_OUT" | cut -d= -f2-)"

if [[ -z "$RUN_DIR" || -z "$GENERATION" || -z "$NUM_TASKS" ]]; then
  echo "Prep did not return RUN_DIR/GENERATION/NUM_TASKS:" >&2
  echo "$PREP_OUT" >&2
  exit 1
fi

echo "Run dir:    $RUN_DIR"
echo "Generation: $GENERATION"
echo "Num tasks:  $NUM_TASKS"

# Phase 2: Submit the jobs array
ARRAY_SPEC="${ARRAY_INDICES:-0-$((NUM_TASKS - 1))}"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,RUN_DIR="$RUN_DIR",GENERATION="$GENERATION" \
  generate/scripts/generate-job.sh)
echo "Submitted array job $ARRAY_JOB_ID (--array=$ARRAY_SPEC)"

# Phase 3: Merge the per-task shards once every task succeeds
FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,RUN_DIR="$RUN_DIR",GENERATION="$GENERATION" \
  generate/scripts/generate-finalize.sh)
echo "Submitted finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
