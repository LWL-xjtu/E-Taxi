#!/usr/bin/env bash
set -euo pipefail

RAW_DATA="${1:-data/raw/yellow_tripdata_2026-01.parquet}"
PROCESSED_DIR="${2:-data/processed/ten15_real}"
TRAIN_DIR="${3:-outputs/ten15_real}"
EVAL_DIR="${4:-outputs/ten15_eval}"

prepare_nyc --config configs/ten_to_fifteen_real.toml --input "${RAW_DATA}" --output "${PROCESSED_DIR}"
train --config configs/ten_to_fifteen_real.toml --data-dir "${PROCESSED_DIR}" --output-dir "${TRAIN_DIR}"

LATEST_CKPT="$(ls -1 "${TRAIN_DIR}"/checkpoints/*.pt | tail -n 1)"
evaluate --config configs/ten_to_fifteen_real.toml --data-dir "${PROCESSED_DIR}" --checkpoint "${LATEST_CKPT}" --output-dir "${EVAL_DIR}" --split test --episodes 1 --stress
