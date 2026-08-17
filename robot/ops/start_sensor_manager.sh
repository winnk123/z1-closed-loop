#!/usr/bin/env bash
set -euo pipefail
# Preserve the standard perception stack, replacing only its incompatible
# UVC RGB-D process with a serial-pinned RealSense bridge for the new D435.
sleep 1
EAME_ROS_ENV=${EAME_ROS_ENV:-/etc/eame/ros2.env}
HT_CAMERA_SETUP=${HT_CAMERA_SETUP:-}
source "$EAME_ROS_ENV"
ros2 run perception_systems ps_sensor_camera_ros2 &
ros2 run perception_systems ps_sync_data_ros2 &
ros2 run perception_systems ps_sensor_fusion_ros2 &
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 "${SCRIPT_DIR}/sensor_manager_realsense_bridge.py" &
if [[ -n "$HT_CAMERA_SETUP" ]]; then
  source "$HT_CAMERA_SETUP"
  ht_camera_driver_node &
fi
