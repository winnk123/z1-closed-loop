#!/usr/bin/env bash
# Start the Z1-local Relay tunnel. This script never starts motion control.
set -euo pipefail

SERVER_HOST="${ULV_SERVER_HOST:?export ULV_SERVER_HOST to the Relay server hostname or IP}"
SERVER_USER="${ULV_SERVER_USER:?export ULV_SERVER_USER to the Relay server SSH user}"
LOCAL_PORT="${ULV_LOCAL_PORT:-18888}"
REMOTE_PORT="${ULV_REMOTE_PORT:-18888}"
KEY_PATH="${ULV_TUNNEL_KEY:-$HOME/.ssh/ulv_relay_ed25519}"
LOG_PATH="${ULV_TUNNEL_LOG:-/tmp/ulv-relay-tunnel.log}"

if [[ ! -r "$KEY_PATH" ]]; then
  echo "missing tunnel key: $KEY_PATH" >&2
  exit 1
fi
if ss -lnt | grep -q ":${LOCAL_PORT}[[:space:]]"; then
  echo "local port ${LOCAL_PORT} is already in use" >&2
  exit 1
fi

nohup ssh -N \
  -i "$KEY_PATH" \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${SERVER_USER}@${SERVER_HOST}" >"$LOG_PATH" 2>&1 &

for _ in $(seq 1 12); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/health"; then
    exit 0
  fi
  sleep 1
done

echo "tunnel did not become healthy; see $LOG_PATH" >&2
exit 1
