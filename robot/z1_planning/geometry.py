"""BBox/depth to temporary camera-aligned local-goal geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .realsense_camera import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class LocalGoal:
    pixel_u: int
    pixel_v: int
    depth_m: float
    x_forward_m: float
    y_left_m: float


def pixel_bbox(
    bbox: Sequence[float | int], *, width: int, height: int, coordinate_space: str
) -> tuple[int, int, int, int]:
    """Validate a bbox and convert the documented Relay coordinate convention."""
    if len(bbox) != 4:
        raise ValueError("bbox_2d must contain exactly four values")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if coordinate_space == "normalized_1000":
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
    elif coordinate_space != "pixels":
        raise ValueError(f"unsupported bbox coordinate space: {coordinate_space}")

    left = int(np.clip(np.floor(min(x1, x2)), 0, width - 1))
    right = int(np.clip(np.ceil(max(x1, x2)), 0, width - 1))
    top = int(np.clip(np.floor(min(y1, y2)), 0, height - 1))
    bottom = int(np.clip(np.ceil(max(y1, y2)), 0, height - 1))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid bbox after conversion: {(left, top, right, bottom)}")
    return left, top, right, bottom


def robust_depth_m(
    depth_mm: np.ndarray,
    *, u: int,
    v: int,
    radius_px: int,
    depth_scale_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> float:
    """Use a small valid-depth median patch instead of a single noisy pixel."""
    if depth_mm.ndim != 2:
        raise ValueError("depth image must be HxW")
    height, width = depth_mm.shape
    u0, u1 = max(0, u - radius_px), min(width, u + radius_px + 1)
    v0, v1 = max(0, v - radius_px), min(height, v + radius_px + 1)
    patch_m = depth_mm[v0:v1, u0:u1].astype(np.float32) * depth_scale_m
    valid = patch_m[(patch_m >= min_depth_m) & (patch_m <= max_depth_m)]
    if valid.size == 0:
        raise ValueError("no valid depth in bbox target patch")
    return float(np.median(valid))


def bbox_to_local_goal(frame: RGBDFrame, bbox: Sequence[float | int], planning: dict) -> tuple[LocalGoal, tuple[int, int, int, int]]:
    """Create a non-executable local goal using the G1 camera-forward convention."""
    if planning.get("coordinate_mode") != "g1_baseline_approx":
        raise ValueError("only g1_baseline_approx is enabled in the zero-motion implementation")
    intrinsics: CameraIntrinsics = frame.intrinsics
    box = pixel_bbox(
        bbox,
        width=intrinsics.width,
        height=intrinsics.height,
        coordinate_space=str(planning["bbox_coordinate_space"]),
    )
    left, _top, right, bottom = box
    u = int(round((left + right) * 0.5))
    v = bottom
    depth_m = robust_depth_m(
        frame.depth_mm,
        u=u,
        v=v,
        radius_px=int(planning["depth_patch_radius_px"]),
        depth_scale_m=frame.depth_scale_m,
        min_depth_m=float(planning["min_depth_m"]),
        max_depth_m=float(planning["max_depth_m"]),
    )
    camera_x_right_m = (u - intrinsics.ppx) * depth_m / intrinsics.fx
    goal_x = min(depth_m, float(planning["max_goal_distance_m"]))
    goal_y = -camera_x_right_m
    return LocalGoal(u, v, depth_m, goal_x, goal_y), box
