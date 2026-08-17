"""Explicitly gated helpers for Z1 high-level gait tests."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class GaitSpec:
    name: str
    sdk_enum_name: str
    confirmation: str


@dataclass(frozen=True)
class RuntimePreflight:
    set_gait_service_available: bool
    robot_hanged: bool | None
    imu_available: bool

    @property
    def allows_execution(self) -> bool:
        return (
            self.set_gait_service_available
            and self.robot_hanged is False
            and self.imu_available
        )


@dataclass(frozen=True)
class JoystickTestSpec:
    """A deliberately small, single-axis joystick test."""

    name: str
    axis_name: str
    default_axis_value: float
    confirmation: str


GAITS = {
    "recovery": GaitSpec(
        name="recovery",
        sdk_enum_name="GAIT_RECOVERY_STAND",
        confirmation="EXECUTE_RECOVERY_STAND",
    ),
    "balance": GaitSpec(
        name="balance",
        sdk_enum_name="GAIT_BALANCE_STAND",
        confirmation="EXECUTE_BALANCE_STAND",
    ),
}

JOYSTICK_TESTS = {
    "forward_stop": JoystickTestSpec(
        name="forward_stop",
        axis_name="left_y_axis",
        default_axis_value=0.05,
        confirmation="EXECUTE_FORWARD_STOP_TEST",
    ),
    "left_turn_stop": JoystickTestSpec(
        name="left_turn_stop",
        axis_name="right_x_axis",
        default_axis_value=-0.05,
        confirmation="EXECUTE_LEFT_TURN_STOP_TEST",
    ),
    "right_turn_stop": JoystickTestSpec(
        name="right_turn_stop",
        axis_name="right_x_axis",
        default_axis_value=0.05,
        confirmation="EXECUTE_RIGHT_TURN_STOP_TEST",
    ),
}


def gait_spec(name: str) -> GaitSpec:
    try:
        return GAITS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported gait: {name}") from exc


def joystick_test_spec(name: str) -> JoystickTestSpec:
    try:
        return JOYSTICK_TESTS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported joystick test: {name}") from exc


def require_execution_consent(*, execute: bool, confirmation: str | None, spec: GaitSpec) -> None:
    if not execute:
        raise PermissionError("refusing to execute gait command without --execute")
    if confirmation != spec.confirmation:
        raise PermissionError(f"confirmation must be exactly: {spec.confirmation}")


def require_runtime_ready(preflight: RuntimePreflight) -> None:
    if not preflight.set_gait_service_available:
        raise RuntimeError("/app/set_gait is not available")
    if preflight.robot_hanged is None:
        raise RuntimeError("/manager/robot_hanged has no fresh sample")
    if preflight.robot_hanged:
        raise RuntimeError("/manager/robot_hanged is true")
    if not preflight.imu_available:
        raise RuntimeError("ROS /imu has no fresh sample")


def inspect_runtime(timeout_s: float) -> RuntimePreflight:
    """Read the ROS prerequisites without importing or calling the Z1 SDK."""
    try:
        import rclpy
        from app_msgs.srv import SetGait
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool
    except ImportError as exc:
        raise RuntimeError(
            "runtime preflight requires the Z1 ROS overlay with app_msgs and rclpy"
        ) from exc

    rclpy.init()
    node = rclpy.create_node("z1_motion_preflight")
    robot_hanged: bool | None = None
    imu_available = False
    service_available = False

    def on_robot_hanged(message: Bool) -> None:
        nonlocal robot_hanged
        robot_hanged = bool(message.data)

    def on_imu(message: Imu) -> None:
        nonlocal imu_available
        imu_available = True

    try:
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        node.create_subscription(Bool, "/manager/robot_hanged", on_robot_hanged, sensor_qos)
        node.create_subscription(Imu, "/imu", on_imu, sensor_qos)
        client = node.create_client(SetGait, "/app/set_gait")
        service_available = client.wait_for_service(timeout_sec=timeout_s)
        deadline = time.monotonic() + timeout_s
        while (robot_hanged is None or not imu_available) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return RuntimePreflight(
        set_gait_service_available=service_available,
        robot_hanged=robot_hanged,
        imu_available=imu_available,
    )


def _require_ok(status: Any, operation: str, sdk: Any) -> None:
    if status.code != sdk.ErrorCode.OK:
        raise RuntimeError(f"{operation} failed: code={status.code}, message={status.message}")


def execute_gait(*, sdk: Any, local_ip: str, spec: GaitSpec, timeout_ms: int) -> dict[str, str | int]:
    """Run one approved high-level gait command and always release the SDK."""
    robot = sdk.MagicRobot()
    connected = False
    try:
        if not robot.initialize(local_ip):
            raise RuntimeError("MagicRobot.initialize returned false")
        _require_ok(robot.connect(), "MagicRobot.connect", sdk)
        connected = True
        controller = robot.get_high_level_motion_controller()
        gait = getattr(sdk.GaitMode, spec.sdk_enum_name)
        _require_ok(controller.set_gait(gait, timeout_ms), f"set_gait({spec.sdk_enum_name})", sdk)
        return {
            "result": "ok",
            "gait": spec.name,
            "sdk_gait": spec.sdk_enum_name,
            "timeout_ms": timeout_ms,
        }
    finally:
        if connected:
            robot.disconnect()
        robot.shutdown()


def execute_recovery_then_balance(
    *, sdk: Any, local_ip: str, timeout_ms: int, recovery_wait_s: float
) -> dict[str, str | int | float]:
    """Apply the vendor-required recovery-stand then balance-stand sequence."""
    if recovery_wait_s < 0.0:
        raise ValueError("recovery_wait_s must be non-negative")
    recovery = gait_spec("recovery")
    balance = gait_spec("balance")
    robot = sdk.MagicRobot()
    connected = False
    try:
        if not robot.initialize(local_ip):
            raise RuntimeError("MagicRobot.initialize returned false")
        _require_ok(robot.connect(), "MagicRobot.connect", sdk)
        connected = True
        controller = robot.get_high_level_motion_controller()
        recovery_gait = getattr(sdk.GaitMode, recovery.sdk_enum_name)
        _require_ok(controller.set_gait(recovery_gait, timeout_ms), f"set_gait({recovery.sdk_enum_name})", sdk)
        time.sleep(recovery_wait_s)
        balance_gait = getattr(sdk.GaitMode, balance.sdk_enum_name)
        _require_ok(controller.set_gait(balance_gait, timeout_ms), f"set_gait({balance.sdk_enum_name})", sdk)
        return {
            "result": "ok",
            "first_gait": recovery.sdk_enum_name,
            "second_gait": balance.sdk_enum_name,
            "recovery_wait_s": recovery_wait_s,
            "timeout_ms": timeout_ms,
        }
    finally:
        if connected:
            robot.disconnect()
        robot.shutdown()


def execute_zero_joystick(*, sdk: Any, local_ip: str) -> dict[str, str]:
    """Send exactly one all-zero high-level joystick command and release the SDK."""
    robot = sdk.MagicRobot()
    connected = False
    try:
        if not robot.initialize(local_ip):
            raise RuntimeError("MagicRobot.initialize returned false")
        _require_ok(robot.connect(), "MagicRobot.connect", sdk)
        connected = True
        command = sdk.JoystickCommand()
        command.left_x_axis = 0.0
        command.left_y_axis = 0.0
        command.right_x_axis = 0.0
        command.right_y_axis = 0.0
        robot.get_high_level_motion_controller().send_joystick_command(command)
        return {"result": "ok", "command": "zero_joystick"}
    finally:
        if connected:
            robot.disconnect()
        robot.shutdown()


def _joystick_command(sdk: Any, *, axis_name: str | None = None, axis_value: float = 0.0) -> Any:
    command = sdk.JoystickCommand()
    command.left_x_axis = 0.0
    command.left_y_axis = 0.0
    command.right_x_axis = 0.0
    command.right_y_axis = 0.0
    if axis_name is not None:
        setattr(command, axis_name, axis_value)
    return command


def _send_joystick(controller: Any, command: Any, sdk: Any) -> None:
    status = controller.send_joystick_command(command)
    if status is not None and hasattr(status, "code"):
        _require_ok(status, "send_joystick_command", sdk)


def _initialize_high_level_controller(robot: Any, sdk: Any) -> Any:
    """Follow the official Python SDK high-level-motion setup sequence."""
    _require_ok(
        robot.set_motion_control_level(sdk.ControllerLevel.HighLevel),
        "set_motion_control_level(HighLevel)",
        sdk,
    )
    controller = robot.get_high_level_motion_controller()
    if not controller.initialize():
        raise RuntimeError("high-level motion controller initialize returned false")
    return controller


def execute_joystick_test(
    *,
    sdk: Any,
    local_ip: str,
    spec: JoystickTestSpec,
    axis_value: float,
    duration_s: float,
    rate_hz: float = 20.0,
    stop_cycles: int = 5,
) -> dict[str, str | float | int]:
    """Send one bounded joystick pulse, then a same-session all-zero stop burst.

    This function deliberately never changes gait. The caller must first verify
    that the robot is already in its approved balance-stand state.
    """
    if not 0.0 < abs(axis_value) <= 0.40:
        raise ValueError("axis_value magnitude must be in (0, 0.40]")
    if not 0.0 < duration_s <= 4.0:
        raise ValueError("duration_s must be in (0, 4.0]")
    if rate_hz != 20.0:
        raise ValueError("rate_hz must be exactly 20.0 for the Z1 SDK motion test")
    if stop_cycles < 3:
        raise ValueError("stop_cycles must be at least 3")

    robot = sdk.MagicRobot()
    connected = False
    controller: Any | None = None
    controller_initialized = False
    nonzero_sent = 0
    zero_sent = 0
    command_started_monotonic_s: float | None = None
    stop_started_monotonic_s: float | None = None
    try:
        if not robot.initialize(local_ip):
            raise RuntimeError("MagicRobot.initialize returned false")
        _require_ok(robot.connect(), "MagicRobot.connect", sdk)
        connected = True
        controller = _initialize_high_level_controller(robot, sdk)
        controller_initialized = True
        motion_command = _joystick_command(
            sdk, axis_name=spec.axis_name, axis_value=axis_value
        )
        period_s = 1.0 / rate_hz
        deadline = time.monotonic() + duration_s
        next_tick = time.monotonic()
        while next_tick < deadline:
            if command_started_monotonic_s is None:
                command_started_monotonic_s = time.monotonic()
            _send_joystick(controller, motion_command, sdk)
            nonzero_sent += 1
            next_tick += period_s
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        # A zero burst is sent for normal completion, errors, and Ctrl-C.
        try:
            if connected and controller is not None:
                stop_started_monotonic_s = time.monotonic()
                zero_command = _joystick_command(sdk)
                for _ in range(stop_cycles):
                    try:
                        _send_joystick(controller, zero_command, sdk)
                        zero_sent += 1
                    finally:
                        time.sleep(1.0 / rate_hz)
        finally:
            if controller_initialized and controller is not None:
                controller.shutdown()
            if connected:
                robot.disconnect()
            robot.shutdown()

    return {
        "result": "ok",
        "test": spec.name,
        "axis_name": spec.axis_name,
        "axis_value": axis_value,
        "duration_s": duration_s,
        "rate_hz": rate_hz,
        "nonzero_command_count": nonzero_sent,
        "zero_command_count": zero_sent,
        "command_started_monotonic_s": command_started_monotonic_s,
        "stop_started_monotonic_s": stop_started_monotonic_s,
    }
