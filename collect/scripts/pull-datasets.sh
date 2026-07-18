#!/bin/bash
#
# Pull dataset folders (and SLURM logs) generated on the HPC cluster down to
# this machine.
#
# Configure once in .env:
#   CLUSTER_SSH_HOST=<~/.ssh/config Host alias for the cluster login node>
#   CLUSTER_REMOTE_DIR=<absolute path to this repo on the cluster>
#
# Usage:
#   ./pull-datasets.sh                       Pull the newest remote dataset into datasets/
#   ./pull-datasets.sh --all                 Pull every remote dataset (mirror datasets/)
#   ./pull-datasets.sh 2026-06-08T14-32-34   Pull one specific dataset by timestamp
#   ./pull-datasets.sh --no-logs             Skip the logs/ directory
#   ./pull-datasets.sh --dry-run             Show what would transfer
#
# Transfers are additive (never deletes local files) and resumable (--partial).

set -euo pipefail

# Absolute path to this script, so usage() works after the cd below
SCRIPT_PATH="$(readlink -f "$0")"

# Run from the repo root so .env, datasets/ and logs/ resolve
cd "$(dirname "$SCRIPT_PATH")/../.."

WANT_ALL=0
WANT_LOGS=1
DRY_RUN=0
DATASET_TIMESTAMP=""
DATASETS_DIR="datasets"

# Prints comments from the top of the script as help text
usage() {
  sed -n '2,17p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) WANT_ALL=1 ;;
    --no-logs) WANT_LOGS=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage 0 ;;
    -*) echo "Unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -n "$DATASET_TIMESTAMP" ]]; then
        echo "Only one dataset timestamp can be given (got '$DATASET_TIMESTAMP' and '$1')." >&2
        exit 1
      fi
      if [[ ! "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Invalid dataset timestamp '$1' (expected YYYY-MM-DDTHH-MM-SS)." >&2
        exit 1
      fi
      DATASET_TIMESTAMP="$1"
      ;;
  esac
  shift
done

if [[ ! -f .env ]]; then
  echo "Missing .env. Add CLUSTER_SSH_HOST and CLUSTER_REMOTE_DIR." >&2
  exit 1
fi

source .env

# Check that required environment variables are set
: "${CLUSTER_SSH_HOST:?Set CLUSTER_SSH_HOST in .env (an ~/.ssh/config Host alias)}"
: "${CLUSTER_REMOTE_DIR:?Set CLUSTER_REMOTE_DIR in .env (the repo path on the cluster)}"

CLUSTER_REMOTE_DIR="${CLUSTER_REMOTE_DIR%/}" # Trim any trailing slash for clean path joins

# Set up rsync options
RSYNC_OPTS=(-az --partial --progress)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(--dry-run)
  echo "[dry-run] no files will be written"
fi

run_rsync() {
  local src="$1" dst="$2"
  echo "+ rsync ${RSYNC_OPTS[*]} -e ssh $src $dst"
  rsync "${RSYNC_OPTS[@]}" -e ssh "$src" "$dst"
}

# Pull all datasets or just one specific dataset
if [[ "$WANT_ALL" -eq 1 ]]; then
  echo "Pulling all datasets from ${CLUSTER_SSH_HOST}:${CLUSTER_REMOTE_DIR}/${DATASETS_DIR}/"
  run_rsync "${CLUSTER_SSH_HOST}:${CLUSTER_REMOTE_DIR}/${DATASETS_DIR}/" \
    "${DATASETS_DIR}/"
else
  if [[ -z "$DATASET_TIMESTAMP" ]]; then
    echo "Finding newest dataset on ${CLUSTER_SSH_HOST}..."
    DATASET_TIMESTAMP="$(ssh "$CLUSTER_SSH_HOST" \
      "cd '$CLUSTER_REMOTE_DIR' && ls -1d ${DATASETS_DIR}/????-??-??T??-??-??/ 2>/dev/null | sort | tail -n1")"
    DATASET_TIMESTAMP="${DATASET_TIMESTAMP#"${DATASETS_DIR}"/}"
    DATASET_TIMESTAMP="${DATASET_TIMESTAMP%/}"
    if [[ -z "$DATASET_TIMESTAMP" ]]; then
      echo "No datasets found under ${CLUSTER_REMOTE_DIR}/${DATASETS_DIR} on the cluster." >&2
      exit 1
    fi
    echo "Newest dataset: $DATASET_TIMESTAMP"
  fi
  run_rsync "${CLUSTER_SSH_HOST}:${CLUSTER_REMOTE_DIR}/${DATASETS_DIR}/${DATASET_TIMESTAMP}/" \
    "${DATASETS_DIR}/${DATASET_TIMESTAMP}/"
fi

# Pull SLURM logs
if [[ "$WANT_LOGS" -eq 1 ]]; then
  echo "Pulling SLURM logs..."
  run_rsync "${CLUSTER_SSH_HOST}:${CLUSTER_REMOTE_DIR}/logs/" "logs/" || {
    status=$?
    # rsync exits 23 when the remote logs/ doesn't exist. Any other exit code is a real failure.
    [[ "$status" -eq 23 ]] || exit "$status"
    echo "(no logs/ on the cluster yet, skipping)"
  }
fi

echo "Done."
