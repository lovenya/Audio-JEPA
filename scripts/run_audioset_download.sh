#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <balanced_train|eval|unbalanced_train> [max_items] [skip_items]"
  exit 1
fi

SPLIT="$1"
MAX_ITEMS="${2:-}"
SKIP_ITEMS="${3:-0}"

DATA_ROOT="${DATA_ROOT:-/scratch/lovenya/datasets/AudioSet}"
WORKERS="${WORKERS:-8}"
RETRIES="${RETRIES:-1}"
TIMEOUT="${TIMEOUT:-120}"

MANIFEST="${DATA_ROOT}/manifests/${SPLIT}.jsonl"
OUT_DIR="${DATA_ROOT}/raw/${SPLIT}"
LOG_DIR="${DATA_ROOT}/logs"
STATUS_LOG="${LOG_DIR}/download_${SPLIT}_max${MAX_ITEMS:-all}_skip${SKIP_ITEMS}.csv"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

CMD=(
  "${DATA_ROOT}/audioset_env/bin/python"
  "scripts/audioset_download.py"
  "--manifest" "${MANIFEST}"
  "--out-dir" "${OUT_DIR}"
  "--yt-dlp-bin" "${DATA_ROOT}/audioset_env/bin/yt-dlp"
  "--workers" "${WORKERS}"
  "--retries" "${RETRIES}"
  "--skip-items" "${SKIP_ITEMS}"
  "--per-clip-timeout" "${TIMEOUT}"
  "--status-log" "${STATUS_LOG}"
)

if [[ -n "${MAX_ITEMS}" ]]; then
  CMD+=("--max-items" "${MAX_ITEMS}")
fi

PYTHONUNBUFFERED=1 "${CMD[@]}"

echo "[done] ${STATUS_LOG}"

