#!/usr/bin/env bash
# Shared environment for Z1 operational scripts.
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
Z1_LOCAL_ENV=${Z1_LOCAL_ENV:-$PROJECT_DIR/.z1_env}
if [[ -r "$Z1_LOCAL_ENV" ]]; then
  source "$Z1_LOCAL_ENV"
fi

EAME_SETUP=${EAME_SETUP:-/opt/eame/setup.bash}
Z1_SDK_ROOT=${Z1_SDK_ROOT:?export Z1_SDK_ROOT=/path/to/magicbot-z1_sdk-main}
MOTION_MSGS_PREFIX=${MOTION_MSGS_PREFIX:-/opt/eame/motion_msgs}

if [[ ! -r "$EAME_SETUP" ]]; then
  echo "missing Z1 setup: $EAME_SETUP" >&2
  return 1 2>/dev/null || exit 1
fi

# The vendor setup references optional variables such as COLCON_TRACE before
# assigning defaults, so it cannot be sourced while the caller has `set -u`.
RESTORE_NOUNSET=false
if [[ $- == *u* ]]; then
  set +u
  RESTORE_NOUNSET=true
fi
source "$EAME_SETUP"
if [[ $RESTORE_NOUNSET == true ]]; then
  set -u
fi
# The Z1 ROS graph is local to the robot.  The vendor setup may export 0,
# which makes this process join a different DDS network from the control stack.
export ROS_DOMAIN_ID=${Z1_ROS_DOMAIN_ID:-2}
export ROS_LOCALHOST_ONLY=${Z1_ROS_LOCALHOST_ONLY:-1}
export RMW_IMPLEMENTATION=${Z1_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export LD_LIBRARY_PATH="$Z1_SDK_ROOT/lib/$(uname -m):$MOTION_MSGS_PREFIX/lib:/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$Z1_SDK_ROOT/lib/$(uname -m)${PYTHONPATH:+:$PYTHONPATH}"
