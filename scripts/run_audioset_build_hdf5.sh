#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <balanced_train|eval|unbalanced_train> [max_items]"
  exit 1
fi

SPLIT="$1"
MAX_ITEMS="${2:-}"

DATA_ROOT="${DATA_ROOT:-/scratch/lovenya/datasets/AudioSet}"
OUT_H5_DIR="${DATA_ROOT}/hdf5"

case "${SPLIT}" in
  balanced_train)
    OUT_H5="${OUT_H5_DIR}/balanced_train_soxrhq.h5"
    OUT_CSV="${OUT_H5_DIR}/silent_files_balanced_train_soxrhq.csv"
    ;;
  eval)
    OUT_H5="${OUT_H5_DIR}/eval_soxrhq.h5"
    OUT_CSV="${OUT_H5_DIR}/silent_files_eval_soxrhq.csv"
    ;;
  unbalanced_train)
    OUT_H5="${OUT_H5_DIR}/full_unbal_bal_train_wav.h5"
    OUT_CSV="${OUT_H5_DIR}/silent_files_full_unbal_bal_train_wav.csv"
    ;;
  *)
    echo "Unknown split: ${SPLIT}"
    exit 1
    ;;
esac

CMD=(
  "ajepa_env/bin/python"
  "scripts/audioset_build_hdf5.py"
  "--manifest" "${DATA_ROOT}/manifests/${SPLIT}.jsonl"
  "--audio-dir" "${DATA_ROOT}/raw/${SPLIT}"
  "--output-h5" "${OUT_H5}"
  "--silent-csv" "${OUT_CSV}"
)

if [[ -n "${MAX_ITEMS}" ]]; then
  CMD+=("--max-items" "${MAX_ITEMS}")
fi

"${CMD[@]}"

ln -sf "${OUT_H5}" "${DATA_ROOT}/$(basename "${OUT_H5}")"
ln -sf "${OUT_CSV}" "${DATA_ROOT}/$(basename "${OUT_CSV}")"

echo "[done] ${OUT_H5}"

