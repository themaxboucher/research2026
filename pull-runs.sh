#!/bin/bash
#
# Pull run folders (and SLURM logs) generated on the HPC cluster down to this
# machine, so dashboard.py / report.py can read them locally.
#
# Configure once in .env:
#   CLUSTER_SSH_HOST=<~/.ssh/config Host alias for the cluster login node>
#   CLUSTER_REMOTE_DIR=<absolute path to this repo on the cluster>
#
# Usage:
#   ./pull-runs.sh                       Pull the newest remote run into runs/
#   ./pull-runs.sh --all                 Pull every remote run (mirror runs/)
#   ./pull-runs.sh 2026-06-08T14-32-34   Pull one specific run by timestamp
#   ./pull-runs.sh --no-logs             Skip the logs/ directory
#   ./pull-runs.sh --dry-run             Show what would transfer, copy nothing
#
# Transfers are additive (never deletes local files) and resumable (--partial).

set -euo pipefail

cd "$(dirname "$0")"

WANT_ALL=0
WANT_LOGS=1
DRY_RUN=0
RUN_TIMESTAMP=""

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) WANT_ALL=1 ;;
    --no-logs) WANT_LOGS=0 ;;
    --logs) WANT_LOGS=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage 0 ;;
    -*) echo "Unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -n "$RUN_TIMESTAMP" ]]; then
        echo "Only one run timestamp can be given (got '$RUN_TIMESTAMP' and '$1')." >&2
        exit 1
      fi
      RUN_TIMESTAMP="$1"
      ;;
  esac
  shift
done

if [[ ! -f .env ]]; then
  echo "Missing .env. Add CLUSTER_SSH_HOST and CLUSTER_REMOTE_DIR." >&2
  exit 1
fi

# Load config from .env without exporting secrets to the wider environment.
set -a
# shellcheck disable=SC1091
source .env
set +a

: "${CLUSTER_SSH_HOST:?Set CLUSTER_SSH_HOST in .env (an ~/.ssh/config Host alias)}"
: "${CLUSTER_REMOTE_DIR:?Set CLUSTER_REMOTE_DIR in .env (the repo path on the cluster)}"

# Trim any trailing slash from the remote dir for clean path joins.
CLUSTER_REMOTE_DIR="${CLUSTER_REMOTE_DIR%/}"

CLUSTER_SCRATCH_DIR="$(dirname "$CLUSTER_REMOTE_DIR")/scratch/$(basename "$CLUSTER_REMOTE_DIR")"

RSYNC_OPTS=(-az --partial --progress)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
  echo "[dry-run] no files will be written"
fi

run_rsync() {
  echo "+ rsync ${RSYNC_OPTS[*]} -e ssh $1 $2"
  rsync "${RSYNC_OPTS[@]}" -e ssh "$1" "$2"
}

mkdir -p runs

if [[ "$WANT_ALL" -eq 1 ]]; then
  echo "Pulling all runs from ${CLUSTER_SSH_HOST}:${CLUSTER_SCRATCH_DIR}/runs/"
  run_rsync "${CLUSTER_SSH_HOST}:${CLUSTER_SCRATCH_DIR}/runs/" "runs/"
else
  if [[ -z "$RUN_TIMESTAMP" ]]; then
    echo "Finding newest run on ${CLUSTER_SSH_HOST}..."
    RUN_TIMESTAMP="$(ssh "$CLUSTER_SSH_HOST" \
      "cd '$CLUSTER_SCRATCH_DIR' && ls -1d runs/*/ 2>/dev/null | sort | tail -n1")"
    RUN_TIMESTAMP="${RUN_TIMESTAMP#runs/}"
    RUN_TIMESTAMP="${RUN_TIMESTAMP%/}"
    if [[ -z "$RUN_TIMESTAMP" ]]; then
      echo "No runs found under ${CLUSTER_SCRATCH_DIR}/runs on the cluster." >&2
      exit 1
    fi
    echo "Newest run: $RUN_TIMESTAMP"
  fi
  # Source has no trailing slash so the timestamp dir itself lands inside runs/.
  run_rsync "${CLUSTER_SSH_HOST}:${CLUSTER_SCRATCH_DIR}/runs/${RUN_TIMESTAMP}" "runs/"
fi

if [[ "$WANT_LOGS" -eq 1 ]]; then
  echo "Pulling SLURM logs..."
  mkdir -p logs
  # Logs may not exist yet on a brand-new checkout; don't fail the whole pull.
  run_rsync "${CLUSTER_SSH_HOST}:${CLUSTER_SCRATCH_DIR}/logs/" "logs/" || \
    echo "(no logs/ on the cluster yet, skipping)"
fi

echo "Done."
