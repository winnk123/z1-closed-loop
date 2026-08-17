#!/usr/bin/env python3
"""Read-only readiness check for the Z1 LA -> VA -> NavDP closed-loop runner.

This script never imports the Z1 SDK, creates no joystick sender, and does not
reset the Relay session.  Its exit code is zero only when every dependency
needed before a physical closed-loop run is currently available.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from z1_planning.config import load_config
from z1_planning.runtime import Z1RosState
from z1_planning.sensor_manager_camera import SensorManagerRGBDCamera


ODOM_TIMEOUT_S = 0.50
IMU_TIMEOUT_S = 0.20


def age_s(now_s: float, timestamp_s: float | None) -> float | None:
    return None if timestamp_s is None else round(now_s - timestamp_s, 4)


def check_runtime(pose_source: str, timeout_s: float) -> dict[str, Any]:
    state: Z1RosState | None = None
    try:
        state = Z1RosState(pose_source=pose_source)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state.spin(wait_timeout_s=0.05)
            if state.odom_t is not None and state.imu_t is not None:
                snapshot = state.snapshot()
                if snapshot.robot_hanged is not None and snapshot.estop is not None and snapshot.robot_fsm is not None:
                    break
        now = time.monotonic()
        snapshot = state.snapshot()
        odom_age = age_s(now, state.odom_t)
        imu_age = age_s(now, state.imu_t)
        localization_ready = pose_source != "lidar_local" or snapshot.localization_healthy is True
        ready = (
            state.odom is not None
            and snapshot.robot_hanged is False
            and snapshot.estop is False
            and snapshot.robot_fsm == 46
            and odom_age is not None and 0.0 <= odom_age <= ODOM_TIMEOUT_S
            and imu_age is not None and 0.0 <= imu_age <= IMU_TIMEOUT_S
            and localization_ready
        )
        return {
            "ready": ready,
            "yaw_and_pose_source": state.pose_topic,
            "robot_hanged": snapshot.robot_hanged,
            "estop": snapshot.estop,
            "robot_fsm": snapshot.robot_fsm,
            "odom_received": state.odom is not None,
            "odom_age_s": odom_age,
            "imu_age_s": imu_age,
            "localization_healthy": snapshot.localization_healthy,
        }
    except Exception as exc:
        return {"ready": False, "error": str(exc)}
    finally:
        if state is not None:
            state.close()


def check_camera(camera_config: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    config = dict(camera_config)
    config["frame_timeout_ms"] = int(timeout_s * 1000.0)
    camera: SensorManagerRGBDCamera | None = None
    try:
        camera = SensorManagerRGBDCamera(config)
        camera.start()
        frame = camera.capture(after_monotonic_s=time.monotonic())
        return {
            "ready": True,
            "color_shape": list(frame.color_bgr.shape),
            "depth_shape": list(frame.depth_mm.shape),
            "intrinsics": {
                "width": frame.intrinsics.width, "height": frame.intrinsics.height,
                "fx": frame.intrinsics.fx, "fy": frame.intrinsics.fy,
            },
        }
    except Exception as exc:
        return {"ready": False, "error": str(exc)}
    finally:
        if camera is not None:
            camera.stop()


def check_relay(relay_config: dict[str, Any], planner_name: str) -> dict[str, Any]:
    url = str(relay_config["url"]).rstrip("/") + "/health"
    timeout_s = float(relay_config.get("timeout_s", 60.0))
    try:
        with urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ready_key = {"navdp": "navdp_ready", "iplanner": "iplanner_ready"}.get(planner_name)
        if ready_key is None:
            raise ValueError("unsupported planner: " + planner_name)
        return {
            "ready": payload.get("ok") is True and payload.get(ready_key) is True,
            "url": url,
            "planner": planner_name,
            "planner_ready": payload.get(ready_key),
            "protocol": payload.get("protocol"),
            "boot_id": payload.get("boot_id"),
        }
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "url": url, "planner": planner_name, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/z1_planning.yaml"))
    parser.add_argument("--pose-source", choices=("chassis", "lidar_local"), default="chassis")
    parser.add_argument("--timeout-s", type=float, default=6.0)
    args = parser.parse_args()
    if not 0.5 <= args.timeout_s <= 30.0 or not math.isfinite(args.timeout_s):
        parser.error("--timeout-s must be finite and in [0.5, 30]")

    config = load_config(args.config)
    planner_name = str(config["planning"].get("planner", "navdp")).lower()
    report = {
        "motion_commanded": False,
        "sdk_sender_created": False,
        "pose_source": args.pose_source,
        "runtime": check_runtime(args.pose_source, args.timeout_s),
        "camera": check_camera(config["camera"], args.timeout_s),
        "relay": check_relay(config["relay"], planner_name),
    }
    report["ready"] = all(report[name].get("ready") is True for name in ("runtime", "camera", "relay"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
