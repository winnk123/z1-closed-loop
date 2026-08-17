#!/usr/bin/env bash
# Execute the optional Baseline-stop-aligned experiment with server NavDP.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$PROJECT_DIR/ops/load_z1_env.sh"
set -u
LOCAL_IP=${1:-${Z1_LOCAL_IP:?export Z1_LOCAL_IP to the robot network interface IP}}
INSTRUCTION=${Z1_INSTRUCTION:-找到桌上的水杯并且停下。}
TARGET_DESCRIPTION=${Z1_TARGET_DESCRIPTION:-桌上的水杯。}

cd "$PROJECT_DIR"
exec python3 scripts/run_z1_lavira_closed_loop.py \
  --execute \
  --enable-motion \
  --confirm=EXECUTE_Z1_LAVIRA_CLOSED_LOOP \
  --local-ip="$LOCAL_IP" \
  --pose-source=chassis \
  --replan-interval-s=2.0 \
  --instruction="$INSTRUCTION" \
  --target-description="$TARGET_DESCRIPTION" \
  --max-duration-s=360 \
  --max-task-steps=12 \
  --artifact-dir=captures/missions/lavira
