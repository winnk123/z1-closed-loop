#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${NAVDP_PYTHON_BIN:-python3}
NAVDP_ROOT=${NAVDP_ROOT:?export NAVDP_ROOT to the NavDP source directory}
NAVDP_CHECKPOINT=${NAVDP_CHECKPOINT:-$NAVDP_ROOT/navdp-cross-modal.ckpt}
HOST=${NAVDP_HOST:-127.0.0.1}
PORT=${NAVDP_PORT:-18889}
DEVICE=${NAVDP_DEVICE:-cuda:0}

[[ -d "$NAVDP_ROOT" ]] || { echo "missing NAVDP_ROOT: $NAVDP_ROOT" >&2; exit 1; }
[[ -f "$NAVDP_CHECKPOINT" ]] || { echo "missing NAVDP_CHECKPOINT: $NAVDP_CHECKPOINT" >&2; exit 1; }

exec "$PYTHON_BIN" "$SCRIPT_DIR/navdp_worker.py" \
  --navdp-root "$NAVDP_ROOT" \
  --checkpoint "$NAVDP_CHECKPOINT" \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE"
