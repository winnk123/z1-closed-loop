"""Pure post-settle yaw correction decisions for chassis-heading tests."""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees

from .strategic_navigation import wrap_angle


@dataclass(frozen=True)
class YawCorrection:
    """The current, signed correction needed to reach one absolute yaw target."""

    residual_rad: float
    direction: str | None
    degrees: float
    within_tolerance: bool


def correction_from_settled_yaw(
    *,
    target_yaw_rad: float,
    settled_yaw_rad: float,
    tolerance_rad: float,
) -> YawCorrection:
    """Describe the next correction using the yaw measured after stopping."""
    residual = wrap_angle(float(target_yaw_rad) - float(settled_yaw_rad))
    magnitude = abs(degrees(residual))
    within_tolerance = abs(residual) <= float(tolerance_rad)
    return YawCorrection(
        residual_rad=residual,
        direction=None if within_tolerance else ("left" if residual > 0.0 else "right"),
        degrees=0.0 if within_tolerance else magnitude,
        within_tolerance=within_tolerance,
    )
