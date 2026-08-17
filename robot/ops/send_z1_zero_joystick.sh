#!/usr/bin/env bash
# Send one all-zero high-level joystick command and exit.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/load_z1_env.sh"
set -u
LOCAL_IP=${1:-${Z1_LOCAL_IP:?export Z1_LOCAL_IP to the robot network interface IP}}

cd "$PROJECT_DIR"
export LOCAL_IP

exec python3 -c '
import json
import os

import magicbot_z1_python as magicbot
from z1_planning.gait import execute_zero_joystick

print(json.dumps(
    execute_zero_joystick(sdk=magicbot, local_ip=os.environ["LOCAL_IP"]),
    indent=2,
))
'
