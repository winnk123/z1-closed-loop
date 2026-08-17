#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${DASHSCOPE_API_KEY:?export DASHSCOPE_API_KEY before starting the Relay}"

PYTHON_BIN=${ULV_PYTHON_BIN:-python3}
HOST=${ULV_RELAY_HOST:-127.0.0.1}
PORT=${ULV_RELAY_PORT:-18888}
MODEL=${ULV_DASHSCOPE_MODEL:-qwen3.8-max}
NAVDP_URL=${ULV_NAVDP_URL:-http://127.0.0.1:18889}

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" ulv_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --dashscope-model "$MODEL" \
  --navdp-url "$NAVDP_URL"
