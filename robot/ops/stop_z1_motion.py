#!/usr/bin/env python3
"""Explicitly gated Z1 all-zero joystick command."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from z1_planning.gait import execute_zero_joystick


CONFIRMATION = "EXECUTE_ZERO_JOYSTICK"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ip", help="SDK local IP. Required only with --execute.")
    parser.add_argument("--execute", action="store_true", help="Permit one physical all-zero command.")
    parser.add_argument("--confirm", help="Exact confirmation string required with --execute.")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({"dry_run": True, "command": "zero_joystick", "motion_commanded": False}, indent=2))
        return 0
    if not args.local_ip:
        parser.error("--local-ip is required with --execute")
    if args.confirm != CONFIRMATION:
        parser.error(f"confirmation must be exactly: {CONFIRMATION}")

    sdk_root = os.environ.get("Z1_SDK_ROOT", "")
    if not sdk_root:
        raise RuntimeError("Z1_SDK_ROOT is required")
    sdk_lib = Path(sdk_root) / "lib" / platform.machine()
    if sdk_lib.is_dir():
        sys.path.insert(0, str(sdk_lib))
        os.environ["LD_LIBRARY_PATH"] = str(sdk_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        ctypes.CDLL(str(sdk_lib / "libmagicbot_z1_sdk.so"), mode=ctypes.RTLD_GLOBAL)
    import magicbot_z1_python as magicbot

    print(json.dumps(execute_zero_joystick(sdk=magicbot, local_ip=args.local_ip), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
