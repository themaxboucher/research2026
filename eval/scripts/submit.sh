#!/bin/bash
#
# Usage:
#   ./submit.sh                             Score the latest run of the latest dataset
#   ./submit.sh --dataset-dir <timestamp>   Evaluate a specific dataset (default: latest)
#   ./submit.sh --run-dir <timestamp>       Evaluate a specific run (default: latest in dataset)
#   ./submit.sh --num-tasks 16              Width of the scoring job array
#   ./submit.sh --array 3,7                 Submit only these task indices (resume)
#   ./submit.sh --skip-setup                Reuse the existing .venv and HF cache; skip uv sync and downloads
#   ./submit.sh --force                     Rescore results that were already scored

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

DATASET_DIR=""
RUN_DIR=""
NUM_TASKS=""
ARRAY_INDICES=""
SKIP_SETUP=""
FORCE=""

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

# Set up the Python environment from uv.lock, with the Python version pinned in
# .python-version. --skip-setup reuses an existing .venv as-is
if [[ -n "$SKIP_SETUP" ]]; then
  if [[ ! -d .venv ]]; then
    echo "No .venv found; run without --skip-setup first to create it." >&2
    exit 1
  fi
else
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null; then
    echo "uv not found; install it once with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  uv sync --locked --no-dev
fi
source .venv/bin/activate

# Warm the shared HF cache from the login node so the offline compute node can
# score. --skip-setup reuses the cache as-is.
if [[ -z "$SKIP_SETUP" ]]; then
  # ARC kills login-node processes that use more than 5 GB of memory, and
  # hf-xet's parallel chunk downloads can go past that. Plain HTTP streams each
  # file straight to disk instead.
  HF_HUB_DISABLE_XET=1 python - <<'EOF'
from huggingface_hub import snapshot_download
import evaluate

from eval.constants import BERTSCORE_MODEL

print(f"Ensuring {BERTSCORE_MODEL} is in the HF cache")
# BERTScore loads the PyTorch weights, so skip the TensorFlow and Flax copies
snapshot_download(BERTSCORE_MODEL, ignore_patterns=["*.h5", "*.msgpack"])

for metric in ("rouge", "bleu", "bertscore"):
    print(f"Caching {metric} metric script")
    evaluate.load(metric)
EOF
fi

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

# Slurm opens each job's --output file in logs/ before the job script runs
mkdir -p logs

# Phase 2: Submit the scoring array
ARRAY_SPEC="${ARRAY_INDICES:-0-$((NUM_TASKS - 1))}"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR",FORCE="$FORCE" \
  eval/scripts/job.sh)
echo "Submitted scoring array $ARRAY_JOB_ID (--array=$ARRAY_SPEC)"

# Phase 3: Merge shards and compute metrics once every task succeeds
FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",RUN_DIR="$RUN_DIR" \
  eval/scripts/finalize.sh)
echo "Submitted finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
