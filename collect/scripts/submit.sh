#!/bin/bash
#
# Usage:
#   ./submit.sh                             Fresh run with defaults
#   ./submit.sh --dataset-dir <timestamp>   Resume an existing dataset (reuses its cache)
#   ./submit.sh --array 3,7,12              Submit only these task indices (resume)
#   ./submit.sh --repos-per-task 10         Repos per task (default 10)
#   ./submit.sh --throttle 20               Max concurrent array tasks (default 20)
#   ./submit.sh --max-repos 1000            Cap repos searched (passed to --prepare)
#   ./submit.sh --repo-min-stars 50         Min stars filter (passed to --prepare)
#   ./submit.sh --skip-setup                Reuse the existing .venv; skip pip install

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

REPOS_PER_TASK=1
THROTTLE=20
DATASET_DIR=""
ARRAY_INDICES=""
MAX_REPOS=""
REPO_MIN_STARS=""
SKIP_SETUP=""

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --repos-per-task) REPOS_PER_TASK="$2"; shift 2 ;;
    --throttle) THROTTLE="$2"; shift 2 ;;
    --max-repos) MAX_REPOS="$2"; shift 2 ;;
    --repo-min-stars) REPO_MIN_STARS="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

if [[ ! -f .env ]]; then
  echo "Missing .env with GITHUB_TOKENS (or GITHUB_TOKEN)." >&2
  exit 1
fi

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

# Phase 1: Prepare
PREP_ARGS=(--repos-per-task "$REPOS_PER_TASK")
[[ -n "$DATASET_DIR" ]] && PREP_ARGS+=(--dataset-dir "$DATASET_DIR")
[[ -n "$MAX_REPOS" ]] && PREP_ARGS+=(--max-repos "$MAX_REPOS")
[[ -n "$REPO_MIN_STARS" ]] && PREP_ARGS+=(--repo-min-stars "$REPO_MIN_STARS")

echo "+ python -m collect.prepare ${PREP_ARGS[*]}"
PREP_OUT="$(python -m collect.prepare "${PREP_ARGS[@]}")"

# Read the printed variables
DATASET_DIR="$(grep '^DATASET_DIR=' <<<"$PREP_OUT" | cut -d= -f2-)"
NUM_TASKS="$(grep '^NUM_TASKS=' <<<"$PREP_OUT" | cut -d= -f2-)"

if [[ -z "$DATASET_DIR" || -z "$NUM_TASKS" ]]; then
  echo "Prep did not return DATASET_DIR/NUM_TASKS:" >&2
  echo "$PREP_OUT" >&2
  exit 1
fi

echo "Dataset dir:   $DATASET_DIR"
echo "Num tasks: $NUM_TASKS"

# Phase 2: Submit the jobs array
ARRAY_SPEC="${ARRAY_INDICES:-0-$((NUM_TASKS - 1))}%${THROTTLE}"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",NUM_TASKS="$NUM_TASKS" \
  collect/scripts/job.sh)
echo "Submitted array job $ARRAY_JOB_ID (--array=$ARRAY_SPEC)"

# Phase 3: Finalize data collection
FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,DATASET_DIR="$DATASET_DIR" \
  collect/scripts/finalize.sh)
echo "Submitted finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
