"""Pure-Python closed-loop follower for a forward/left ground path.

The follower only produces a desired velocity request.  A safety bridge owns
the enable/disable decision and the SDK mapping, so importing this module is
safe on a development machine and on the robot without ROS or SDK packages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, cos, hypot, isfinite, pi, sin
from typing import Any, Sequence


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class OdomPose:
    """Robot pose in the path frame (x=forward, y=left, yaw from +x)."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class PathFollowerConfig:
    """Conservative defaults suitable for the first safety-bridge integration."""

    control_period_s: float = 0.05  # intended 20 Hz
    lookahead_m: float = 0.45
    cruise_speed_mps: float = 0.10
    max_forward_speed_mps: float = 0.10
    max_yaw_speed_radps: float = 0.35
    goal_tolerance_m: float = 0.12
    max_lateral_error_m: float = 0.75
    max_heading_error_rad: float = pi / 2.0
    max_duration_s: float = 60.0
    odom_timeout_s: float = 0.30

    def __post_init__(self) -> None:
        for name in ("control_period_s", "lookahead_m", "cruise_speed_mps",
                     "max_forward_speed_mps", "max_yaw_speed_radps",
                     "goal_tolerance_m", "max_lateral_error_m",
                     "max_duration_s", "odom_timeout_s"):
            if not isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isfinite(float(self.max_heading_error_rad)) or not 0 < self.max_heading_error_rad <= pi:
            raise ValueError("max_heading_error_rad must be in (0, pi]")


@dataclass(frozen=True)
class PathFollowerOutput:
    """One command-sized output record for the safety bridge."""

    sequence: int
    timestamp_s: float
    vx_mps: float
    vyaw_radps: float
    active: bool
    stop_reason: str | None
    progress_m: float
    goal_distance_m: float
    lateral_error_m: float
    heading_error_rad: float
    target_index: int

    @property
    def should_send(self) -> bool:
        """Whether a non-zero request is safe for the bridge to consider."""
        return self.active and self.stop_reason is None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"should_send": self.should_send}


def _normalise_path(path: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    if len(path) < 2:
        raise ValueError("path must contain at least two points")
    result: list[tuple[float, float]] = []
    for point in path:
        if len(point) < 2:
            raise ValueError("each path point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("path contains non-finite coordinates")
        if not result or hypot(x - result[-1][0], y - result[-1][1]) > 1e-6:
            result.append((x, y))
    if len(result) < 2:
        raise ValueError("path must contain a non-zero segment")
    return tuple(result)


class PathFollower:
    """Stateful 20 Hz pure-pursuit follower with monotonic path progress."""

    def __init__(self, path: Sequence[Sequence[float]], config: PathFollowerConfig | None = None) -> None:
        self.path = _normalise_path(path)
        self.config = config or PathFollowerConfig()
        self._cumulative = [0.0]
        for a, b in zip(self.path, self.path[1:]):
            self._cumulative.append(self._cumulative[-1] + hypot(b[0] - a[0], b[1] - a[1]))
        self._total = self._cumulative[-1]
        self._progress = 0.0
        self._sequence = 0
        self._last_output_timestamp = float("-inf")
        self._start_timestamp: float | None = None
        self._last_odom_timestamp: float | None = None
        self._stopped_reason: str | None = None

    @property
    def done(self) -> bool:
        return self._stopped_reason is not None

    @property
    def progress_m(self) -> float:
        return self._progress

    def _output(self, timestamp_s: float, *, vx: float = 0.0, vyaw: float = 0.0,
                active: bool = False, stop_reason: str | None = None,
                goal_distance: float = float("inf"), lateral: float = 0.0,
                heading: float = 0.0, target_index: int = 0) -> PathFollowerOutput:
        # Preserve strictly monotonic timestamps even if an upstream message repeats a stamp.
        timestamp_s = max(float(timestamp_s), self._last_output_timestamp + 1e-6)
        self._last_output_timestamp = timestamp_s
        self._sequence += 1
        return PathFollowerOutput(self._sequence, timestamp_s, float(vx), float(vyaw), active,
                                  stop_reason, self._progress, goal_distance, lateral, heading,
                                  target_index)

    def _stop(self, timestamp_s: float, reason: str, **kwargs: Any) -> PathFollowerOutput:
        self._stopped_reason = reason
        return self._output(timestamp_s, active=False, stop_reason=reason, **kwargs)

    def _point_at(self, distance_m: float) -> tuple[float, float, int]:
        distance_m = max(0.0, min(distance_m, self._total))
        for index in range(len(self.path) - 1):
            length = self._cumulative[index + 1] - self._cumulative[index]
            if distance_m <= self._cumulative[index + 1] or index == len(self.path) - 2:
                f = (distance_m - self._cumulative[index]) / length
                a, b = self.path[index], self.path[index + 1]
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), index
        return self.path[-1][0], self.path[-1][1], len(self.path) - 2

    def _project_progress(self, pose: OdomPose) -> tuple[float, float, float, int]:
        best: tuple[float, float, float, float, int] | None = None
        for index, (a, b) in enumerate(zip(self.path, self.path[1:])):
            segment_start = self._cumulative[index]
            segment_length = self._cumulative[index + 1] - segment_start
            if self._cumulative[index + 1] < self._progress:
                continue
            # Ignore path behind the monotonic progress point.
            lower = max(0.0, self._progress - segment_start)
            dx, dy = b[0] - a[0], b[1] - a[1]
            raw = ((pose.x_m - a[0]) * dx + (pose.y_m - a[1]) * dy) / (segment_length * segment_length)
            fraction = max(lower / segment_length, min(1.0, raw))
            px, py = a[0] + fraction * dx, a[1] + fraction * dy
            distance = hypot(pose.x_m - px, pose.y_m - py)
            candidate = (distance, segment_start + fraction * segment_length, px, py, index)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        distance, projected_s, px, py, index = best
        a, b = self.path[index], self.path[index + 1]
        tangent = atan2(b[1] - a[1], b[0] - a[0])
        signed_lateral = -sin(tangent) * (pose.x_m - px) + cos(tangent) * (pose.y_m - py)
        self._progress = max(self._progress, projected_s)
        return self._progress, distance, signed_lateral, index

    def update(self, pose: OdomPose, timestamp_s: float) -> PathFollowerOutput:
        """Consume one odometry sample and return the next desired request."""
        timestamp_s = float(timestamp_s)
        if not all(isfinite(float(value)) for value in (pose.x_m, pose.y_m, pose.yaw_rad, timestamp_s)):
            return self._stop(timestamp_s, "invalid_odom")
        if self._stopped_reason is not None:
            return self._output(timestamp_s, stop_reason=self._stopped_reason)
        if self._start_timestamp is None:
            self._start_timestamp = timestamp_s
        if self._last_odom_timestamp is not None and timestamp_s < self._last_odom_timestamp:
            return self._stop(timestamp_s, "timestamp_regression")
        if (self._last_odom_timestamp is not None
                and timestamp_s - self._last_odom_timestamp > self.config.odom_timeout_s):
            return self._stop(timestamp_s, "odom_stale")
        self._last_odom_timestamp = timestamp_s
        if timestamp_s - self._start_timestamp > self.config.max_duration_s:
            return self._stop(timestamp_s, "timeout")

        progress, _cross_track_distance, lateral, segment_index = self._project_progress(pose)
        goal_x, goal_y, _ = self._point_at(self._total)
        goal_distance = hypot(goal_x - pose.x_m, goal_y - pose.y_m)
        if goal_distance <= self.config.goal_tolerance_m:
            return self._stop(timestamp_s, "goal_reached", goal_distance=goal_distance,
                              lateral=lateral, target_index=len(self.path) - 1)
        if abs(lateral) > self.config.max_lateral_error_m:
            return self._stop(timestamp_s, "lateral_error", goal_distance=goal_distance,
                              lateral=lateral, target_index=segment_index)

        lookahead_x, lookahead_y, target_index = self._point_at(min(self._total, progress + self.config.lookahead_m))
        target_heading = atan2(lookahead_y - pose.y_m, lookahead_x - pose.x_m)
        heading_error = _wrap_angle(target_heading - pose.yaw_rad)
        if abs(heading_error) > self.config.max_heading_error_rad:
            return self._stop(timestamp_s, "heading_error", goal_distance=goal_distance,
                              lateral=lateral, heading=heading_error, target_index=target_index)

        target_distance = max(hypot(lookahead_x - pose.x_m, lookahead_y - pose.y_m), 0.10)
        heading_factor = max(0.0, 1.0 - abs(heading_error) / self.config.max_heading_error_rad)
        speed = min(self.config.cruise_speed_mps, self.config.max_forward_speed_mps) * heading_factor
        yaw_rate = max(-self.config.max_yaw_speed_radps,
                       min(self.config.max_yaw_speed_radps, 2.0 * speed * sin(heading_error) / target_distance))
        return self._output(timestamp_s, vx=speed, vyaw=yaw_rate, active=True,
                            goal_distance=goal_distance, lateral=lateral,
                            heading=heading_error, target_index=target_index)

    def step(self, x_m: float, y_m: float, yaw_rad: float, timestamp_s: float) -> PathFollowerOutput:
        """Convenience form for callers that have scalar odometry fields."""
        return self.update(OdomPose(float(x_m), float(y_m), float(yaw_rad)), timestamp_s)


__all__ = ["OdomPose", "PathFollower", "PathFollowerConfig", "PathFollowerOutput"]
