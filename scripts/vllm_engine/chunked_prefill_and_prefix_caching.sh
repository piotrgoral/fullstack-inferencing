#!/usr/bin/env bash
# Combined server-side knobs: chunked prefill + prefix caching (no speculative decoding).
# Good "combined" arm if you are not running a draft model.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=/dev/null
source ./_common.sh
exec vllm serve "$VLLM_MODEL" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --host "$VLLM_BIND_HOST" \
  --port "$VLLM_SERVE_PORT" \
  --enable-chunked-prefill \
  --enable-prefix-caching
