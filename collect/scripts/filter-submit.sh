#!/bin/bash
#
# Usage:
#   ./filter-submit.sh                             Filter the latest dataset
#   ./filter-submit.sh --dataset-dir <timestamp>   Filter a specific dataset (default: latest)
#   ./filter-submit.sh --num-tasks 100             Width of the filtering job array (default 100)
#   ./filter-submit.sh --array 3,7                 Submit only these task indices (resume)
#   ./filter-submit.sh --throttle 8                Max concurrent array tasks (default: unthrottled)
#   ./filter-submit.sh --no-finalize               Skip the dependent merge job
#
# --num-tasks partitions the commits, so a resume with --array has to repeat the
# --num-tasks the dataset was first filtered with. Finalize refuses to merge
# shards that disagree.

set -euo pipefail

# Run everything from the repo root
cd "$(dirname "$0")/../.."

DATASET_DIR=""
NUM_TASKS=100
ARRAY_INDICES=""
THROTTLE=""
NO_FINALIZE=""

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --array) ARRAY_INDICES="$2"; shift 2 ;;
    --throttle) THROTTLE="$2"; shift 2 ;;
    --no-finalize) NO_FINALIZE=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
done

if [[ ! "$NUM_TASKS" =~ ^[0-9]+$ ]] || (( NUM_TASKS < 1 )); then
  echo "--num-tasks must be a positive integer, got '$NUM_TASKS'" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "No .venv found; run ./collect/scripts/submit.sh first to create it." >&2
  exit 1
fi

mkdir -p logs

# Phase 1: Submit the filtering array. Each task owns a hash-partitioned share
# of the commits and writes its own dataset and manifest shards.
ARRAY_SPEC="${ARRAY_INDICES:-0-$((NUM_TASKS - 1))}"
[[ -n "$THROTTLE" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"

ARRAY_JOB_ID=$(sbatch --parsable \
  --array="$ARRAY_SPEC" \
  --export=ALL,DATASET_DIR="$DATASET_DIR",NUM_TASKS="$NUM_TASKS" \
  collect/scripts/filter.sh)
echo "Submitted filtering array $ARRAY_JOB_ID (--array=$ARRAY_SPEC)"

# Phase 2: Merge the shards and sum their manifests once every task succeeds
if [[ -n "$NO_FINALIZE" ]]; then
  echo "Skipping the finalize job (--no-finalize)"
  exit 0
fi

FINALIZE_JOB_ID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JOB_ID" \
  --export=ALL,DATASET_DIR="$DATASET_DIR" \
  collect/scripts/filter-finalize.sh)
echo "Submitted filter finalize job $FINALIZE_JOB_ID (afterok:$ARRAY_JOB_ID)"
