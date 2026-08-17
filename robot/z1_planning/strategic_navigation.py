"""Pure Baseline-compatible LA decision and heading-turn primitives.

The strategic model chooses a discrete heading from a panorama.  This module
deliberately contains no HTTP, ROS, camera, or SDK code: callers must validate
the LA response before it is permitted to influence a physical turn.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any


_DIRECTION_YAW_RAD = {
    "front": 0.0,
    "left": math.pi / 2.0,
    "right": -math.pi / 2.0,
    "behind": math.pi,
}


def wrap_angle(angle_rad: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class StrategicDecision:
    """Validated subset of a Baseline Language-Action response."""

    action: str
    direction: str | None
    selected_view: str | None
    turn_degrees: int
    expected_landmark: str
    progress_analysis: str
    reasoning: str
    todo_list: str

    @property
    def should_stop(self) -> bool:
        return self.action == "STOP"


def parse_strategic_decision(
    payload: str | dict[str, Any], *, selectable_views: set[str] | None = None,
) -> StrategicDecision:
    """Parse exactly the LA contract used by the Baseline navigation loop.

    Unknown action/direction values are rejected rather than silently mapped to
    forward.  This is essential because the value subsequently controls a body
    rotation rather than a simulator action.
    """
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("LA response is not valid JSON") from exc
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("LA response must be a JSON object or JSON string")
    action = str(data.get("action", "")).strip().upper()
    if action not in {"NAVIGATE", "STOP"}:
        raise ValueError("LA action must be NAVIGATE or STOP")
    raw_direction = data.get("turn_direction", data.get("direction"))
    direction = str(raw_direction).strip().lower() if raw_direction is not None else None
    selected_view: str | None = None
    turn_degrees = 0
    if action == "STOP":
        # G1 still turns to LA's nominated direction and calls VA once for the
        # final visual verification.  A STOP response therefore retains the
        # legacy direction field; selected_view is optional for this contract.
        if direction not in _DIRECTION_YAW_RAD:
            direction = "front"
        turn_degrees = 0 if direction == "front" else 180 if direction == "behind" else 90
    elif selectable_views is not None:
        raw_view = data.get("selected_view")
        selected_view = str(raw_view).strip().lower() if raw_view is not None else ""
        if selected_view not in selectable_views:
            raise ValueError("LA NAVIGATE response requires one offered selected_view")
        match = re.fullmatch(r"(front|left|right)_(\d{3})deg", selected_view)
        if match is None:
            raise ValueError("selected_view must encode a physical direction and angle")
        selected_direction, raw_degrees = match.groups()
        # The model chooses an observed image, never a commanded turn.  Decode
        # the physical rotation exclusively from that selected image label.
        direction = selected_direction
        turn_degrees = int(raw_degrees)
        if (direction == "front" and turn_degrees != 0) or (
            direction in {"left", "right"} and turn_degrees not in {30, 60, 90}
        ):
            raise ValueError("selected_view has an unsupported turn angle")
    else:
        if action == "NAVIGATE" and direction not in _DIRECTION_YAW_RAD:
            raise ValueError("LA NAVIGATE response requires front, left, right, or behind")
        turn_degrees = 0 if direction == "front" else 180 if direction == "behind" else 90
    return StrategicDecision(
        action=action,
        direction=direction,
        selected_view=selected_view,
        turn_degrees=turn_degrees,
        expected_landmark=str(data.get("expected_landmark", "")).strip(),
        progress_analysis=str(data.get("progress_analysis", "")).strip(),
        reasoning=str(data.get("reasoning", "")).strip(),
        todo_list=str(data.get("updated_todo_list", "")).strip(),
    )


@dataclass(frozen=True)
class HeadingTurnConfig:
    """Conservative closed-loop yaw controller configuration."""

    yaw_tolerance_rad: float = math.radians(4.0)
    proportional_gain: float = 0.8
    max_yaw_rate_radps: float = 0.12
    slowdown_error_rad: float = math.radians(10.0)
    near_target_max_yaw_rate_radps: float = 0.025
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        if not 0.0 < self.yaw_tolerance_rad < math.pi / 2.0:
            raise ValueError("yaw_tolerance_rad must be in (0, pi/2)")
        if self.proportional_gain <= 0.0 or self.max_yaw_rate_radps <= 0.0 or self.timeout_s <= 0.0:
            raise ValueError("heading-turn parameters must be positive")
        if not self.yaw_tolerance_rad < self.slowdown_error_rad < math.pi / 2.0:
            raise ValueError("slowdown_error_rad must exceed yaw_tolerance_rad and be below pi/2")
        if not 0.0 < self.near_target_max_yaw_rate_radps <= self.max_yaw_rate_radps:
            raise ValueError("near_target_max_yaw_rate_radps must be in (0, max_yaw_rate_radps]")


@dataclass(frozen=True)
class HeadingTurnCommand:
    target_yaw_rad: float
    yaw_error_rad: float
    desired_yaw_rate_radps: float
    complete: bool
    failure_reason: str | None = None


def _tapered_yaw_rate(error_rad: float, config: HeadingTurnConfig) -> float:
    """Use progressively lower yaw authority near the target to limit coast-over."""
    error = float(error_rad)
    requested = config.proportional_gain * error
    magnitude = abs(error)
    limit = config.max_yaw_rate_radps
    if magnitude < config.slowdown_error_rad:
        ratio = magnitude / config.slowdown_error_rad
        # Quadratic taper retains decisive gross turns but makes the final
        # several degrees deliberately slow enough for the physical body.
        limit = config.near_target_max_yaw_rate_radps + (
            config.max_yaw_rate_radps - config.near_target_max_yaw_rate_radps
        ) * ratio * ratio
    return max(-limit, min(requested, limit))


class HeadingTurnController:
    """Track one LA-selected relative heading using current odometry yaw."""

    def __init__(self, initial_yaw_rad: float, direction: str, started_monotonic_s: float,
                 config: HeadingTurnConfig | None = None) -> None:
        if direction not in _DIRECTION_YAW_RAD:
            raise ValueError(f"unsupported strategic direction: {direction}")
        self.config = config or HeadingTurnConfig()
        self.direction = direction
        self.started_monotonic_s = float(started_monotonic_s)
        self.target_yaw_rad = wrap_angle(float(initial_yaw_rad) + _DIRECTION_YAW_RAD[direction])

    def update(self, current_yaw_rad: float, now_monotonic_s: float) -> HeadingTurnCommand:
        now = float(now_monotonic_s)
        error = wrap_angle(self.target_yaw_rad - float(current_yaw_rad))
        if now - self.started_monotonic_s > self.config.timeout_s:
            return HeadingTurnCommand(self.target_yaw_rad, error, 0.0, False, "heading_turn_timeout")
        if abs(error) <= self.config.yaw_tolerance_rad:
            return HeadingTurnCommand(self.target_yaw_rad, error, 0.0, True)
        return HeadingTurnCommand(self.target_yaw_rad, error, _tapered_yaw_rate(error, self.config), False)


@dataclass(frozen=True)
class DirectedTurnCommand:
    """A commanded signed relative turn, retaining direction through 180 degrees."""

    target_yaw_rad: float
    achieved_yaw_rad: float
    yaw_error_rad: float
    desired_yaw_rate_radps: float
    complete: bool
    failure_reason: str | None = None


class DirectedTurnController:
    """Track a left/right relative angle without a shortest-path ambiguity at 180 degrees."""

    def __init__(
        self,
        initial_yaw_rad: float,
        direction: str,
        degrees: float,
        started_monotonic_s: float,
        config: HeadingTurnConfig | None = None,
    ) -> None:
        if direction not in {"left", "right"}:
            raise ValueError("direction must be left or right")
        if not 1.0 <= float(degrees) <= 180.0:
            raise ValueError("degrees must be in [1, 180]")
        self.config = config or HeadingTurnConfig()
        self.direction = direction
        self.started_monotonic_s = float(started_monotonic_s)
        sign = 1.0 if direction == "left" else -1.0
        self.target_yaw_rad = wrap_angle(float(initial_yaw_rad) + sign * math.radians(float(degrees)))
        self._target_signed_rad = sign * math.radians(float(degrees))
        self._previous_yaw_rad = float(initial_yaw_rad)
        self._achieved_signed_rad = 0.0

    def update(self, current_yaw_rad: float, now_monotonic_s: float) -> DirectedTurnCommand:
        current = float(current_yaw_rad)
        delta = wrap_angle(current - self._previous_yaw_rad)
        self._achieved_signed_rad += delta
        self._previous_yaw_rad = current
        error = self._target_signed_rad - self._achieved_signed_rad
        now = float(now_monotonic_s)
        if now - self.started_monotonic_s > self.config.timeout_s:
            return DirectedTurnCommand(
                self.target_yaw_rad, self._achieved_signed_rad, error, 0.0, False, "heading_turn_timeout",
            )
        if abs(error) <= self.config.yaw_tolerance_rad:
            return DirectedTurnCommand(self.target_yaw_rad, self._achieved_signed_rad, error, 0.0, True)
        return DirectedTurnCommand(
            self.target_yaw_rad, self._achieved_signed_rad, error, _tapered_yaw_rate(error, self.config), False,
        )


@dataclass
class TurnLedgerEntry:
    """One LA-directed turn and the measured angle that must be reversed."""

    direction: str
    requested_degrees: float
    measured_signed_degrees: float
    start_yaw_rad: float
    final_yaw_rad: float
    return_direction: str
    return_degrees: float
    returned_signed_degrees: float | None = None

    @property
    def returned(self) -> bool:
        return self.returned_signed_degrees is not None

    def as_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "direction": self.direction,
            "requested_degrees": self.requested_degrees,
            "measured_signed_degrees": self.measured_signed_degrees,
            "start_yaw_rad": self.start_yaw_rad,
            "final_yaw_rad": self.final_yaw_rad,
            "return_direction": self.return_direction,
            "return_degrees": self.return_degrees,
            "returned_signed_degrees": self.returned_signed_degrees,
            "returned": self.returned,
        }


class TurnLedger:
    """Persistent LA turn memory; a return is derived only from measured rotation."""

    def __init__(self) -> None:
        self.entries: list[TurnLedgerEntry] = []

    def record_turn(
        self,
        *,
        direction: str,
        requested_degrees: float,
        measured_signed_degrees: float,
        start_yaw_rad: float,
        final_yaw_rad: float,
    ) -> TurnLedgerEntry:
        if direction not in {"left", "right"}:
            raise ValueError("direction must be left or right")
        values = (requested_degrees, measured_signed_degrees, start_yaw_rad, final_yaw_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("turn ledger values must be finite")
        if not 1.0 <= abs(float(measured_signed_degrees)) <= 180.0:
            raise ValueError("measured turn must be in [1, 180] degrees")
        expected_sign = 1.0 if direction == "left" else -1.0
        if float(measured_signed_degrees) * expected_sign <= 0.0:
            raise ValueError("measured turn sign does not match direction")
        return_direction = "right" if direction == "left" else "left"
        entry = TurnLedgerEntry(
            direction=direction,
            requested_degrees=float(requested_degrees),
            measured_signed_degrees=float(measured_signed_degrees),
            start_yaw_rad=float(start_yaw_rad),
            final_yaw_rad=float(final_yaw_rad),
            return_direction=return_direction,
            return_degrees=abs(float(measured_signed_degrees)),
        )
        self.entries.append(entry)
        return entry

    def pending_return(self) -> TurnLedgerEntry:
        for entry in reversed(self.entries):
            if not entry.returned:
                return entry
        raise RuntimeError("turn ledger has no pending return")

    def record_return(self, measured_signed_degrees: float) -> TurnLedgerEntry:
        entry = self.pending_return()
        measured = float(measured_signed_degrees)
        if not math.isfinite(measured) or measured == 0.0:
            raise ValueError("returned turn must be finite and non-zero")
        expected_sign = -1.0 if entry.direction == "left" else 1.0
        if measured * expected_sign <= 0.0:
            raise ValueError("returned turn sign does not reverse source turn")
        entry.returned_signed_degrees = measured
        return entry

    def as_dict(self) -> dict[str, object]:
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "pending_returns": sum(not entry.returned for entry in self.entries),
        }
