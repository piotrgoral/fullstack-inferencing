#!/usr/bin/env bash
set -euo pipefail

# Local sweep for q35_4b_base-conv experiments.
# Matches yesterday's 5 configurations from data/experiments_q35_4b_base-conv-*.csv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export EXPERIMENT_PREFIX="q3_4b_ngram-2"
export NUM_CONVERSATIONS=64
export MAX_CONVERSATIONS=64

# Update if today's gateway log file name changes.
LOG_PATH="logs/gateway/gateway_metrics_2026-05-24.jsonl"

# Each entry: "<MAX_TURNS_PER_CONV> <MAX_SESSIONS> <MAX_TOKENS>"
configs=(
  "16 16 1024"
  "16 32 1024"
  "16 64 1024"
  "16 64 2048"
  "2 32 1024"
)

for cfg in "${configs[@]}"; do
  read -r MAX_TURNS_PER_CONV MAX_SESSIONS MAX_TOKENS <<<"${cfg}"

  export MAX_TURNS_PER_CONV MAX_SESSIONS MAX_TOKENS

  export TECHNIQUE="${EXPERIMENT_PREFIX}-num-conv-${NUM_CONVERSATIONS}-max-turns-${MAX_TURNS_PER_CONV}-max-sessions-${MAX_SESSIONS}-max-tokens-${MAX_TOKENS}"

  echo "=== Running ${TECHNIQUE} ==="

  python sharegpt_turn_bench.py \
    --technique "${TECHNIQUE}" \
    --mode static \
    --num-conversations "${NUM_CONVERSATIONS}" \
    --max-conversations "${MAX_CONVERSATIONS}" \
    --max-turns-per-conv "${MAX_TURNS_PER_CONV}" \
    --max-sessions "${MAX_SESSIONS}" \
    --max-tokens "${MAX_TOKENS}"

  python scripts/export_experiments.py \
    --log-path "${LOG_PATH}" \
    --technique "${TECHNIQUE}" \
    --output-csv "data/experiments_${TECHNIQUE}.csv"

  echo "=== Finished ${TECHNIQUE} ==="
  echo
done

