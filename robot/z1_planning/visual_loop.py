"""Pure geometry and admission checks for rolling visual navigation."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite, sin
from typing import Sequence

from .path_follower import OdomPose


def _points(path: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in path:
        if len(point) < 2:
            raise ValueError("trajectory point must contain forward and left coordinates")
        x, y = float(point[0]), float(point[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("trajectory contains non-finite coordinates")
        if not result or hypot(x - result[-1][0], y - result[-1][1]) > 1e-6:
            result.append((x, y))
    if len(result) < 2:
        raise ValueError("trajectory must contain a non-zero segment")
    return result


def truncate_path(path: Sequence[Sequence[float]], length_m: float) -> list[list[float]]:
    """Return the prefix ending exactly at ``length_m`` of a local path."""
    if not isfinite(length_m) or length_m <= 0.0:
        raise ValueError("length_m must be finite and positive")
    points = _points(path)
    remaining = float(length_m)
    output = [list(points[0])]
    for first, second in zip(points, points[1:]):
        segment = hypot(second[0] - first[0], second[1] - first[1])
        if remaining <= segment:
            ratio = remaining / segment
            output.append([first[0] + ratio * (second[0] - first[0]),
                           first[1] + ratio * (second[1] - first[1])])
            return output
        output.append(list(second))
        remaining -= segment
    return output


def path_length(path: Sequence[Sequence[float]]) -> float:
    """Return planar arc length for a forward/left NavDP trajectory."""
    points = _points(path)
    return sum(hypot(second[0] - first[0], second[1] - first[1])
               for first, second in zip(points, points[1:]))


def truncate_path_for_stop_margin(path: Sequence[Sequence[float]], stop_margin_m: float) -> list[list[float]]:
    """Apply Baseline's final-distance reserve before a path reaches control.

    The Go1 Baseline does not move when the full planned arc is at or below
    ``SAFE_DISTANCE``.  Mirroring that behavior here is safer than reducing a
    waypoint's X coordinate, which would break curved NavDP trajectories.
    """
    if not isfinite(float(stop_margin_m)) or float(stop_margin_m) < 0.0:
        raise ValueError("stop_margin_m must be finite and non-negative")
    total = path_length(path)
    execute_length = total - float(stop_margin_m)
    if execute_length <= 0.0:
        raise ValueError(
            f"planned path {total:.3f}m is not longer than stop margin {float(stop_margin_m):.3f}m"
        )
    return truncate_path(path, execute_length)


def local_path_to_odom(path: Sequence[Sequence[float]], origin: OdomPose) -> list[list[float]]:
    """Transform forward/left camera-frame path points into the odom XY frame."""
    points = _points(path)
    c, s = cos(origin.yaw_rad), sin(origin.yaw_rad)
    return [[origin.x_m + c * forward - s * left, origin.y_m + s * forward + c * left]
            for forward, left in points]


def reserve_baseline_execution_goal(
    path: Sequence[Sequence[float]], *, origin: OdomPose, safe_distance_m: float,
) -> tuple[list[list[float]], tuple[float, float]]:
    """Freeze G1's one-time safety-truncated endpoint in the odom frame.

    G1 removes SAFE_DISTANCE once from the initial planner path. Its controller
    then replans to that retained endpoint; it does not shorten every replan.
    """
    reserved_path = truncate_path_for_stop_margin(path, safe_distance_m)
    odom_path = local_path_to_odom(reserved_path, origin)
    end = odom_path[-1]
    return reserved_path, (end[0], end[1])


def odom_goal_to_local(goal_x_m: float, goal_y_m: float, origin: OdomPose) -> tuple[float, float]:
    """Express an odom-frame target as forward/left at ``origin``."""
    dx, dy = float(goal_x_m) - origin.x_m, float(goal_y_m) - origin.y_m
    c, s = cos(origin.yaw_rad), sin(origin.yaw_rad)
    return c * dx + s * dy, -s * dx + c * dy


def constrain_replan_to_fixed_goal(
    path: Sequence[Sequence[float]], *, origin: OdomPose, goal_odom: tuple[float, float],
    max_terminal_error_m: float,
) -> tuple[list[list[float]], float]:
    """Require a rolling NavDP path to terminate at the reserved odom goal.

    NavDP only receives the retained endpoint as a point-goal condition and can
    return a trajectory which misses it.  Do not hand that unconstrained path
    to the physical follower.  A close terminal prediction is retained, but
    its final point is replaced with the exact fixed endpoint.
    """
    if not isfinite(float(max_terminal_error_m)) or float(max_terminal_error_m) <= 0.0:
        raise ValueError("max_terminal_error_m must be finite and positive")
    local_path = [list(point) for point in _points(path)]
    terminal_odom = local_path_to_odom(local_path, origin)[-1]
    terminal_error = hypot(terminal_odom[0] - float(goal_odom[0]), terminal_odom[1] - float(goal_odom[1]))
    if terminal_error > float(max_terminal_error_m):
        raise ValueError(
            f"replan endpoint is {terminal_error:.3f}m from fixed goal; limit is {float(max_terminal_error_m):.3f}m"
        )
    goal_forward, goal_left = odom_goal_to_local(*goal_odom, origin)
    local_path[-1] = [goal_forward, goal_left]
    return local_path, terminal_error


def baseline_goal_reached(
    pose: OdomPose, goal_odom: tuple[float, float], *, goal_tolerance_m: float,
) -> bool:
    """Match G1's distance check against the fixed truncated goal, not a replan endpoint."""
    if not isfinite(float(goal_tolerance_m)) or float(goal_tolerance_m) <= 0.0:
        raise ValueError("goal_tolerance_m must be finite and positive")
    return hypot(float(goal_odom[0]) - pose.x_m, float(goal_odom[1]) - pose.y_m) < float(goal_tolerance_m)


@dataclass(frozen=True)
class VisualLoopConfig:
    """Conservative admission limits for a receding-horizon visual controller."""

    horizon_m: float = 0.30
    max_plan_age_s: float = 1.00
    max_total_travel_m: float = 1.00
    max_consecutive_plan_failures: int = 2

    def __post_init__(self) -> None:
        for name in ("horizon_m", "max_plan_age_s", "max_total_travel_m"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.max_consecutive_plan_failures) < 1:
            raise ValueError("max_consecutive_plan_failures must be positive")


@dataclass(frozen=True)
class BaselineTravelPolicy:
    """The two distinct G1 thresholds used by one strategic subgoal."""

    safe_distance_m: float = 0.50
    goal_tolerance_m: float = 1.00

    def __post_init__(self) -> None:
        for name in ("safe_distance_m", "goal_tolerance_m"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class VisualPlan:
    local_path: list[list[float]]
    odom_path: list[list[float]]
    captured_monotonic_s: float
    origin: OdomPose

    def is_fresh(self, now_s: float, max_age_s: float) -> bool:
        age = float(now_s) - self.captured_monotonic_s
        return isfinite(age) and 0.0 <= age <= max_age_s


def admit_visual_plan(
    trajectory: Sequence[Sequence[float]], *, origin: OdomPose, captured_monotonic_s: float,
    config: VisualLoopConfig,
) -> VisualPlan:
    """Validate and shorten a visual plan before the follower can consume it."""
    if not isfinite(float(captured_monotonic_s)):
        raise ValueError("captured_monotonic_s must be finite")
    local_path = truncate_path(trajectory, config.horizon_m)
    return VisualPlan(
        local_path=local_path,
        odom_path=local_path_to_odom(local_path, origin),
        captured_monotonic_s=float(captured_monotonic_s),
        origin=origin,
    )


def admit_baseline_replan(
    trajectory: Sequence[Sequence[float]], *, origin: OdomPose, captured_monotonic_s: float,
) -> VisualPlan:
    """Admit a full replan to G1's already-reserved local execution goal."""
    if not isfinite(float(captured_monotonic_s)):
        raise ValueError("captured_monotonic_s must be finite")
    local_path = [list(point) for point in _points(trajectory)]
    return VisualPlan(
        local_path=local_path,
        odom_path=local_path_to_odom(local_path, origin),
        captured_monotonic_s=float(captured_monotonic_s),
        origin=origin,
    )


__all__ = ["BaselineTravelPolicy", "VisualLoopConfig", "VisualPlan", "admit_baseline_replan", "admit_visual_plan",
           "baseline_goal_reached", "constrain_replan_to_fixed_goal", "local_path_to_odom", "odom_goal_to_local", "path_length", "reserve_baseline_execution_goal", "truncate_path",
           "truncate_path_for_stop_margin"]
