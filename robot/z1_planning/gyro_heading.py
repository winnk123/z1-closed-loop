"""Relative heading control from robot IMU angular velocity only."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GyroHeadingState:
    """Task-relative yaw derived only from debiased IMU gyro_z samples."""

    heading_rad: float
    gyro_bias_rps: float
    timestamp_s: float | None
    valid: bool
    failure_reason: str | None = None

    @property
    def heading_degrees(self) -> float:
        return math.degrees(self.heading_rad)


class GyroHeadingTracker:
    """Maintain a task-local heading whose origin is explicitly zero.

    The tracker has no global-heading claim.  Call :meth:`start` after the
    robot is stationary and its gyro bias has been measured; subsequent IMU
    samples update the relative heading until a sample gap invalidates it.
    """

    def __init__(self, *, gyro_bias_rps: float, max_sample_gap_s: float = 0.10) -> None:
        if not math.isfinite(float(gyro_bias_rps)):
            raise ValueError("gyro bias must be finite")
        if not 0.0 < float(max_sample_gap_s) <= 1.0:
            raise ValueError("max sample gap must be in (0, 1] seconds")
        self._gyro_bias_rps = float(gyro_bias_rps)
        self._max_sample_gap_s = float(max_sample_gap_s)
        self._heading_rad = 0.0
        self._last_timestamp_s: float | None = None
        self._last_rate_rps: float | None = None
        self._failure_reason: str | None = None

    def start(self, *, gyro_timestamp_s: float, gyro_z_rps: float) -> GyroHeadingState:
        """Set the task origin to exactly zero at a fresh IMU sample."""
        timestamp, raw_rate = float(gyro_timestamp_s), float(gyro_z_rps)
        if not all(math.isfinite(value) for value in (timestamp, raw_rate)):
            self._failure_reason = "invalid_gyro_sample"
            return self.snapshot()
        self._heading_rad = 0.0
        self._last_timestamp_s = timestamp
        self._last_rate_rps = raw_rate - self._gyro_bias_rps
        self._failure_reason = None
        return self.snapshot()

    def update(self, *, gyro_timestamp_s: float, gyro_z_rps: float) -> GyroHeadingState:
        timestamp, raw_rate = float(gyro_timestamp_s), float(gyro_z_rps)
        if self._failure_reason is not None:
            return self.snapshot()
        if not all(math.isfinite(value) for value in (timestamp, raw_rate)):
            self._failure_reason = "invalid_gyro_sample"
            return self.snapshot()
        corrected_rate = raw_rate - self._gyro_bias_rps
        if self._last_timestamp_s is None:
            self._last_timestamp_s, self._last_rate_rps = timestamp, corrected_rate
            return self.snapshot()
        if timestamp < self._last_timestamp_s:
            self._failure_reason = "gyro_timestamp_rollback"
            return self.snapshot()
        if timestamp == self._last_timestamp_s:
            return self.snapshot()
        dt = timestamp - self._last_timestamp_s
        if dt > self._max_sample_gap_s:
            self._failure_reason = "gyro_sample_gap"
            return self.snapshot()
        self._heading_rad += 0.5 * (self._last_rate_rps + corrected_rate) * dt
        self._last_timestamp_s, self._last_rate_rps = timestamp, corrected_rate
        return self.snapshot()

    def snapshot(self) -> GyroHeadingState:
        return GyroHeadingState(
            heading_rad=self._heading_rad,
            gyro_bias_rps=self._gyro_bias_rps,
            timestamp_s=self._last_timestamp_s,
            valid=self._failure_reason is None and self._last_timestamp_s is not None,
            failure_reason=self._failure_reason,
        )


@dataclass(frozen=True)
class GyroHeadingConfig:
    yaw_tolerance_rad: float = math.radians(1.5)
    slowdown_error_rad: float = math.radians(10.0)
    max_yaw_rate_radps: float = 0.12
    near_target_yaw_rate_radps: float = 0.08
    timeout_s: float = 45.0

    def __post_init__(self) -> None:
        if not 0.0 < self.yaw_tolerance_rad < self.slowdown_error_rad < math.pi / 2.0:
            raise ValueError("gyro heading tolerances must satisfy 0 < tolerance < slowdown < pi/2")
        if not 0.0 < self.near_target_yaw_rate_radps <= self.max_yaw_rate_radps:
            raise ValueError("near-target rate must be in (0, max rate]")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True)
class GyroHeadingCommand:
    achieved_signed_rad: float
    error_rad: float
    desired_yaw_rate_radps: float
    complete: bool
    failure_reason: str | None = None


class GyroRelativeTurnController:
    """Integrate debiased ``gyro_z`` for one bounded left/right turn."""

    def __init__(
        self,
        *,
        direction: str,
        degrees: float,
        gyro_bias_rps: float,
        started_monotonic_s: float,
        config: GyroHeadingConfig | None = None,
    ) -> None:
        if direction not in {"left", "right"}:
            raise ValueError("direction must be left or right")
        if not 1.0 <= float(degrees) <= 90.0:
            raise ValueError("degrees must be in [1, 90]")
        self.config = config or GyroHeadingConfig()
        self._sign = 1.0 if direction == "left" else -1.0
        self._target_signed_rad = self._sign * math.radians(float(degrees))
        self._bias_rps = float(gyro_bias_rps)
        self._started_s = float(started_monotonic_s)
        self._last_sample_t: float | None = None
        self._last_rate_rps: float | None = None
        self._achieved_signed_rad = 0.0

    @property
    def achieved_signed_rad(self) -> float:
        return self._achieved_signed_rad

    def update(self, *, gyro_timestamp_s: float, gyro_z_rps: float, now_s: float) -> GyroHeadingCommand:
        sample_t, rate, now = float(gyro_timestamp_s), float(gyro_z_rps), float(now_s)
        if not all(math.isfinite(value) for value in (sample_t, rate, now)):
            return GyroHeadingCommand(self._achieved_signed_rad, math.nan, 0.0, False, "invalid_gyro_sample")
        corrected_rate = rate - self._bias_rps
        if self._last_sample_t is None:
            self._last_sample_t, self._last_rate_rps = sample_t, corrected_rate
        elif sample_t > self._last_sample_t:
            dt = sample_t - self._last_sample_t
            if dt > 0.10:
                return GyroHeadingCommand(self._achieved_signed_rad, math.nan, 0.0, False, "gyro_sample_gap")
            self._achieved_signed_rad += 0.5 * (self._last_rate_rps + corrected_rate) * dt
            self._last_sample_t, self._last_rate_rps = sample_t, corrected_rate
        error = self._target_signed_rad - self._achieved_signed_rad
        if now - self._started_s > self.config.timeout_s:
            return GyroHeadingCommand(self._achieved_signed_rad, error, 0.0, False, "gyro_heading_timeout")
        if abs(error) <= self.config.yaw_tolerance_rad:
            return GyroHeadingCommand(self._achieved_signed_rad, error, 0.0, True)
        magnitude = abs(error)
        if magnitude >= self.config.slowdown_error_rad:
            rate_limit = self.config.max_yaw_rate_radps
        else:
            ratio = magnitude / self.config.slowdown_error_rad
            rate_limit = self.config.near_target_yaw_rate_radps + (
                self.config.max_yaw_rate_radps - self.config.near_target_yaw_rate_radps
            ) * ratio
        return GyroHeadingCommand(
            self._achieved_signed_rad,
            error,
            math.copysign(rate_limit, error),
            False,
        )
