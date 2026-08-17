#!/usr/bin/env bash
# Physically run recovery stand followed by balance stand on the Z1.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/load_z1_env.sh"

IGNORE_HANGED=()
if [[ ${1:-} == "--ignore-hanged" ]]; then
  IGNORE_HANGED=(--ignore-hanged)
  shift
fi
LOCAL_IP=${1:-${Z1_LOCAL_IP:?export Z1_LOCAL_IP or pass the robot network interface IP as the first argument}}

exec python3 "$SCRIPT_DIR/start_z1_balance_stand.py" \
  --local-ip "$LOCAL_IP" \
  --execute \
  --confirm=EXECUTE_RECOVERY_THEN_BALANCE \
  "${IGNORE_HANGED[@]}"
