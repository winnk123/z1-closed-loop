"""Fail-closed Z1 motion bridge primitives.

This module deliberately contains no ROS, Z1 SDK, or network code.  A runtime
adapter may feed :class:`Z1SafetyBridge.evaluate` at a fixed rate and map the
returned axes to ``JoystickCommand``.  The bridge is disabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Z1SafetyBridgeConfig:
    """Limits and watchdogs for the high-level Z1 joystick interface.

    ``forward_mps_per_axis`` is intentionally explicit: Z1 joystick values are
    normalized inputs, not m/s.  The default reflects the observed calibration
    (axis 0.30 produced about 0.10 m/s), and must be revalidated per robot.
    Yaw has no confirmed physical calibration, so its mapping is a conservative
    commissioning value rather than an SDK guarantee.
    """

    command_timeout_s: float = 0.20
    odom_timeout_s: float = 0.20
    imu_timeout_s: float = 0.20
    localization_timeout_s: float = 0.20
    expected_robot_fsm: int = 46
    max_forward_speed_mps: float = 0.10
    max_reverse_speed_mps: float = 0.0
    max_abs_yaw_rate_rps: float = 0.20
    forward_mps_per_axis: float = 1.0 / 3.0
    yaw_rps_per_axis: float = 0.50
    max_axis_abs: float = 0.40
    max_command_accel_mps2: float = 0.10
    max_command_yaw_accel_rps2: float = 0.20
    max_axis_rate_per_s: float = 0.40
    future_stamp_tolerance_s: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            "command_timeout_s", "odom_timeout_s", "imu_timeout_s", "localization_timeout_s",
            "max_forward_speed_mps", "forward_mps_per_axis",
            "yaw_rps_per_axis", "max_axis_abs", "max_command_accel_mps2",
            "max_command_yaw_accel_rps2", "max_axis_rate_per_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_reverse_speed_mps < 0.0 or self.future_stamp_tolerance_s < 0.0:
            raise ValueError("reverse speed and future tolerance must be non-negative")
        if self.max_axis_abs > 1.0:
            raise ValueError("max_axis_abs cannot exceed Z1 joystick range 1.0")
        if self.max_forward_speed_mps / self.forward_mps_per_axis > self.max_axis_abs:
            raise ValueError("forward speed limit maps above max_axis_abs")
        if self.max_abs_yaw_rate_rps / self.yaw_rps_per_axis > self.max_axis_abs:
            raise ValueError("yaw rate limit maps above max_axis_abs")


@dataclass(frozen=True)
class Z1RuntimeState:
    """Snapshot of the minimum state required before output is enabled."""

    robot_hanged: bool | None
    estop: bool | None
    robot_fsm: int | None
    odom_timestamp_s: float | None
    imu_timestamp_s: float | None
    localization_healthy: bool | None = None
    localization_timestamp_s: float | None = None

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Z1RuntimeState":
        """Accept both the SDK/ROS ``emy_stop`` spelling and ``estop``."""
        estop = values.get("estop", values.get("emy_stop"))
        return cls(
            robot_hanged=values.get("robot_hanged"),
            estop=estop,
            robot_fsm=values.get("robot_fsm"),
            odom_timestamp_s=values.get("odom_timestamp_s", values.get("odom_t_s")),
            imu_timestamp_s=values.get("imu_timestamp_s", values.get("imu_t_s")),
            localization_healthy=values.get("localization_healthy"),
            localization_timestamp_s=values.get("localization_timestamp_s"),
        )


@dataclass(frozen=True)
class Z1JoystickAxes:
    left_x_axis: float = 0.0
    left_y_axis: float = 0.0
    right_x_axis: float = 0.0
    right_y_axis: float = 0.0

    @classmethod
    def zero(cls) -> "Z1JoystickAxes":
        return cls()


@dataclass(frozen=True)
class Z1BridgeOutput:
    axes: Z1JoystickAxes
    active: bool
    stop_reason: str
    requested_vx_mps: float
    requested_vyaw_rps: float
    applied_vx_mps: float
    applied_vyaw_rps: float
    command_age_s: float
    odom_age_s: float
    imu_age_s: float

    @property
    def left_x_axis(self) -> float:
        return self.axes.left_x_axis

    @property
    def left_y_axis(self) -> float:
        return self.axes.left_y_axis

    @property
    def right_x_axis(self) -> float:
        return self.axes.right_x_axis

    @property
    def right_y_axis(self) -> float:
        return self.axes.right_y_axis


class Z1SafetyBridge:
    """Pure stateful safety gate and command mapper.

    Call ``evaluate`` at 20 Hz or faster.  Every failed gate returns four zero
    axes and resets the slew state.  SDK failures must be reported through
    ``latch_sdk_error``; clearing the latch is an explicit operator action.
    """

    def __init__(self, config: Z1SafetyBridgeConfig | None = None) -> None:
        self.config = config or Z1SafetyBridgeConfig()
        self._sdk_error: str | None = None
        self._last_eval_s: float | None = None
        self._last_command_stamp_s: float | None = None
        self._last_vx = 0.0
        self._last_vyaw = 0.0
        self._last_axes = Z1JoystickAxes.zero()

    @property
    def sdk_error_latched(self) -> bool:
        return self._sdk_error is not None

    @property
    def sdk_error_reason(self) -> str | None:
        return self._sdk_error

    def latch_sdk_error(self, reason: str) -> None:
        self._sdk_error = reason.strip() or "sdk_error"

    def clear_sdk_error(self) -> None:
        """Clear only after the caller has independently verified the SDK."""
        self._sdk_error = None

    @staticmethod
    def _age(now_s: float, stamp_s: float | None) -> float:
        try:
            stamp = float(stamp_s) if stamp_s is not None else math.nan
        except (TypeError, ValueError):
            return math.inf
        if not math.isfinite(stamp):
            return math.inf
        return float(now_s) - stamp

    def _stop(self, reason: str, vx: float, vyaw: float, now_s: float,
              command_age: float = math.inf, odom_age: float = math.inf,
              imu_age: float = math.inf) -> Z1BridgeOutput:
        self._last_eval_s = now_s
        self._last_vx = self._last_vyaw = 0.0
        self._last_axes = Z1JoystickAxes.zero()
        return Z1BridgeOutput(
            axes=Z1JoystickAxes.zero(), active=False, stop_reason=reason,
            requested_vx_mps=vx, requested_vyaw_rps=vyaw,
            applied_vx_mps=0.0, applied_vyaw_rps=0.0,
            command_age_s=command_age, odom_age_s=odom_age, imu_age_s=imu_age,
        )

    def evaluate(
        self,
        *,
        desired_vx_mps: float,
        desired_vyaw_rps: float,
        now_s: float,
        command_timestamp_s: float | None,
        runtime_state: Z1RuntimeState | dict[str, Any] | None,
        motion_enabled: bool = False,
    ) -> Z1BridgeOutput:
        """Evaluate one command; no call here can actuate the robot."""
        vx, vyaw = float(desired_vx_mps), float(desired_vyaw_rps)
        now = float(now_s)
        if not all(math.isfinite(value) for value in (vx, vyaw, now)):
            return self._stop("invalid_command", vx, vyaw, now)
        if self._last_eval_s is not None and now <= self._last_eval_s:
            return self._stop("non_monotonic_time", vx, vyaw, now)
        if not motion_enabled:
            return self._stop("motion_disabled", vx, vyaw, now)
        if self._sdk_error is not None:
            return self._stop("sdk_error_latched:" + self._sdk_error, vx, vyaw, now)
        if command_timestamp_s is None or not math.isfinite(float(command_timestamp_s)):
            return self._stop("command_missing", vx, vyaw, now)
        command_stamp = float(command_timestamp_s)
        command_age = now - command_stamp
        if command_age < -self.config.future_stamp_tolerance_s:
            return self._stop("command_timestamp_in_future", vx, vyaw, now, command_age)
        if command_age > self.config.command_timeout_s:
            return self._stop("command_timeout", vx, vyaw, now, command_age)
        if self._last_command_stamp_s is not None and command_stamp <= self._last_command_stamp_s:
            return self._stop("command_out_of_order", vx, vyaw, now, command_age)
        self._last_command_stamp_s = command_stamp
        if runtime_state is None:
            return self._stop("runtime_state_missing", vx, vyaw, now, command_age)
        state = (Z1RuntimeState.from_mapping(runtime_state)
                 if isinstance(runtime_state, dict) else runtime_state)
        if not isinstance(state, Z1RuntimeState):
            return self._stop("runtime_state_invalid", vx, vyaw, now, command_age)
        odom_age = self._age(now, state.odom_timestamp_s)
        imu_age = self._age(now, state.imu_timestamp_s)
        localization_age = self._age(now, state.localization_timestamp_s)
        if state.robot_hanged is not False:
            return self._stop("robot_hanged_unknown_or_true", vx, vyaw, now, command_age, odom_age, imu_age)
        if state.estop is not False:
            return self._stop("emergency_stop_unknown_or_true", vx, vyaw, now, command_age, odom_age, imu_age)
        if state.robot_fsm != self.config.expected_robot_fsm:
            return self._stop("fsm_not_balance_stand", vx, vyaw, now, command_age, odom_age, imu_age)
        if odom_age < 0.0 or odom_age > self.config.odom_timeout_s:
            return self._stop("odom_stale", vx, vyaw, now, command_age, odom_age, imu_age)
        if imu_age < 0.0 or imu_age > self.config.imu_timeout_s:
            return self._stop("imu_stale", vx, vyaw, now, command_age, odom_age, imu_age)
        if state.localization_healthy is not None or state.localization_timestamp_s is not None:
            if state.localization_healthy is not True:
                return self._stop("localization_unhealthy", vx, vyaw, now, command_age, odom_age, imu_age)
            if localization_age < 0.0 or localization_age > self.config.localization_timeout_s:
                return self._stop("localization_stale", vx, vyaw, now, command_age, odom_age, imu_age)

        dt = self.config.command_timeout_s if self._last_eval_s is None else now - self._last_eval_s
        if dt <= 0.0:
            return self._stop("non_monotonic_time", vx, vyaw, now, command_age, odom_age, imu_age)
        target_vx = min(max(vx, -self.config.max_reverse_speed_mps), self.config.max_forward_speed_mps)
        target_vyaw = max(-self.config.max_abs_yaw_rate_rps, min(vyaw, self.config.max_abs_yaw_rate_rps))
        self._last_vx = self._slew(self._last_vx, target_vx, self.config.max_command_accel_mps2 * dt)
        self._last_vyaw = self._slew(self._last_vyaw, target_vyaw, self.config.max_command_yaw_accel_rps2 * dt)
        # Physical verification: Z1 SDK left_y positive drives forward.
        # Keep the logical forward convention aligned with the SDK edge.
        # /odom_chassis positive yaw is left (CCW); Z1 SDK right_x positive is right.
        target_axes = Z1JoystickAxes(
            right_x_axis=-self._last_vyaw / self.config.yaw_rps_per_axis,
            left_y_axis=self._last_vx / self.config.forward_mps_per_axis,
        )
        max_delta = self.config.max_axis_rate_per_s * dt
        axes = Z1JoystickAxes(
            left_x_axis=self._slew(self._last_axes.left_x_axis, target_axes.left_x_axis, max_delta),
            left_y_axis=self._slew(self._last_axes.left_y_axis, target_axes.left_y_axis, max_delta),
            right_x_axis=self._slew(self._last_axes.right_x_axis, target_axes.right_x_axis, max_delta),
            right_y_axis=self._slew(self._last_axes.right_y_axis, target_axes.right_y_axis, max_delta),
        )
        self._last_axes = axes
        self._last_eval_s = now
        return Z1BridgeOutput(
            axes=axes, active=True, stop_reason="", requested_vx_mps=vx,
            requested_vyaw_rps=vyaw, applied_vx_mps=self._last_vx,
            applied_vyaw_rps=self._last_vyaw, command_age_s=command_age,
            odom_age_s=odom_age, imu_age_s=imu_age,
        )

    @staticmethod
    def _slew(previous: float, target: float, maximum_delta: float) -> float:
        return max(previous - maximum_delta, min(target, previous + maximum_delta))
