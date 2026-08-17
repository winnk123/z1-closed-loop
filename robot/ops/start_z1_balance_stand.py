#!/usr/bin/env python3
"""Run the Z1 recovery-stand (1) then balance-stand (46) sequence."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from z1_planning.gait import execute_recovery_then_balance


CONFIRMATION = "EXECUTE_RECOVERY_THEN_BALANCE"


@dataclass(frozen=True)
class GaitRuntimeState:
    robot_hanged: bool | None
    robot_fsm: int | None


def read_runtime_state(timeout_s: float) -> GaitRuntimeState:
    """Read only the state needed to guard and verify a gait transition."""
    try:
        import rclpy
        from motion_msgs.msg import State
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool
    except ImportError as exc:
        raise RuntimeError("Z1 ROS overlay with motion_msgs is required") from exc

    rclpy.init()
    node = rclpy.create_node("z1_recovery_balance_stand")
    hanged: bool | None = None
    fsm: int | None = None

    def on_hanged(message: Bool) -> None:
        nonlocal hanged
        hanged = bool(message.data)

    def on_leg_state(message: State) -> None:
        nonlocal fsm
        fsm = int(message.robot_fsm)

    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    try:
        node.create_subscription(Bool, "/manager/robot_hanged", on_hanged, qos)
        node.create_subscription(State, "/leg_state", on_leg_state, qos)
        deadline = time.monotonic() + timeout_s
        while (hanged is None or fsm is None) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return GaitRuntimeState(hanged, fsm)


def configure_sdk_import() -> None:
    sdk_root = os.environ.get("Z1_SDK_ROOT", "")
    if not sdk_root:
        raise RuntimeError("Z1_SDK_ROOT is required")
    sdk_lib = Path(sdk_root) / "lib" / platform.machine()
    if not sdk_lib.is_dir():
        raise RuntimeError(f"Z1 SDK library directory does not exist: {sdk_lib}")
    sys.path.insert(0, str(sdk_lib))
    os.environ["LD_LIBRARY_PATH"] = str(sdk_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ip", default=os.environ.get("Z1_LOCAL_IP"))
    parser.add_argument("--recovery-wait-s", type=float, default=10.0)
    parser.add_argument("--runtime-timeout-s", type=float, default=6.0)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ignore-hanged", action="store_true",
                        help="operator override: run the gait sequence even when robot_hanged is not false")
    parser.add_argument("--confirm", help=f"exact confirmation: {CONFIRMATION}")
    args = parser.parse_args()
    if args.execute and not args.local_ip:
        parser.error("--local-ip or Z1_LOCAL_IP is required with --execute")
    if not 0.0 <= args.recovery_wait_s <= 30.0:
        parser.error("--recovery-wait-s must be in [0, 30]")
    if not 0.1 <= args.runtime_timeout_s <= 10.0:
        parser.error("--runtime-timeout-s must be in [0.1, 10]")
    if not 1 <= args.timeout_ms <= 10000:
        parser.error("--timeout-ms must be in [1, 10000]")

    if not args.execute and not args.check_runtime:
        print(json.dumps({
            "dry_run": True,
            "motion_commanded": False,
            "sequence": ["GAIT_RECOVERY_STAND=1", "GAIT_BALANCE_STAND=46"],
            "recovery_wait_s": args.recovery_wait_s,
            "required_confirmation": CONFIRMATION,
        }, indent=2))
        return 0

    before = read_runtime_state(args.runtime_timeout_s)
    if args.check_runtime and not args.execute:
        print(json.dumps({"runtime": asdict(before), "allows_execution": before.robot_hanged is False}, indent=2))
        return 0 if before.robot_hanged is False else 1
    if args.confirm != CONFIRMATION:
        parser.error(f"confirmation must be exactly: {CONFIRMATION}")
    if before.robot_hanged is not False and not args.ignore_hanged:
        raise RuntimeError(f"refusing gait sequence: robot_hanged={before.robot_hanged!r}")

    configure_sdk_import()
    import magicbot_z1_python as magicbot

    result = execute_recovery_then_balance(
        sdk=magicbot,
        local_ip=args.local_ip,
        timeout_ms=args.timeout_ms,
        recovery_wait_s=args.recovery_wait_s,
    )
    after = read_runtime_state(args.runtime_timeout_s)
    report = {"motion_commanded": True, "ignore_hanged": args.ignore_hanged,
              "before": asdict(before), **result, "after": asdict(after)}
    print(json.dumps(report, indent=2))
    if after.robot_fsm != 46:
        raise RuntimeError(f"balance stand verification failed: robot_fsm={after.robot_fsm!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
