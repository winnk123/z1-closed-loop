"""Single-D435 RGB-D capture selected by immutable RealSense serial number."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class RGBDFrame:
    """Color and depth are pixel-aligned and share the color intrinsics."""

    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale_m: float
    captured_monotonic_s: float


class RealSenseRGBDCamera:
    """Own one RealSense pipeline and never select a camera by /dev/video index."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._rs: Any | None = None
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._depth_scale_m: float | None = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # keeps offline/unit-test imports possible
            raise RuntimeError("pyrealsense2 is required on the Z1 Jetson") from exc

        serial = str(self.config["serial"])
        found = {
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        }
        if serial not in found:
            raise RuntimeError(f"RealSense serial {serial!r} is not connected; found={sorted(found)}")

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(serial)
        # The chest D435 is on USB 2.1 and has been validated with bgr8.
        rs_config.enable_stream(
            rs.stream.color,
            int(self.config["color_width"]),
            int(self.config["color_height"]),
            rs.format.bgr8,
            int(self.config["fps"]),
        )
        rs_config.enable_stream(
            rs.stream.depth,
            int(self.config["depth_width"]),
            int(self.config["depth_height"]),
            rs.format.z16,
            int(self.config["fps"]),
        )

        try:
            profile = pipeline.start(rs_config)
        except Exception as exc:
            raise RuntimeError(f"failed to start D435 {serial}: {exc}") from exc

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        self._intrinsics = CameraIntrinsics(
            width=color_profile.width(),
            height=color_profile.height(),
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
        )
        self._depth_scale_m = float(depth_sensor.get_depth_scale())

        try:
            for _ in range(int(self.config.get("warmup_frames", 30))):
                pipeline.wait_for_frames(int(self.config.get("frame_timeout_ms", 5000)))
        except Exception:
            self.stop()
            raise

    def capture(self) -> RGBDFrame:
        if not self._pipeline or not self._align or not self._intrinsics or self._depth_scale_m is None:
            raise RuntimeError("camera is not started")
        try:
            frames = self._pipeline.wait_for_frames(int(self.config.get("frame_timeout_ms", 5000)))
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
        except Exception as exc:
            raise RuntimeError(f"D435 frame acquisition failed: {exc}") from exc
        if not color_frame or not depth_frame:
            raise RuntimeError("D435 returned an incomplete aligned RGB-D frame")

        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_mm = np.asanyarray(depth_frame.get_data()).copy()
        if color_bgr.shape[:2] != depth_mm.shape[:2]:
            raise RuntimeError(
                f"aligned frame size mismatch: color={color_bgr.shape}, depth={depth_mm.shape}"
            )
        rotation = int(self.config.get("image_rotation_deg", 0))
        color_bgr, depth_mm, intrinsics = self._rotate_aligned_frame(
            color_bgr, depth_mm, self._intrinsics, rotation
        )
        return RGBDFrame(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            intrinsics=intrinsics,
            depth_scale_m=self._depth_scale_m,
            captured_monotonic_s=time.monotonic(),
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None

    @staticmethod
    def _rotate_aligned_frame(
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics: CameraIntrinsics,
        rotation_deg: int,
    ) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """Rotate RGB and aligned depth identically, including virtual intrinsics."""
        if rotation_deg == 0:
            return color_bgr, depth_mm, intrinsics
        if rotation_deg == 90:
            return (
                np.rot90(color_bgr, k=3).copy(),
                np.rot90(depth_mm, k=3).copy(),
                CameraIntrinsics(
                    width=intrinsics.height,
                    height=intrinsics.width,
                    fx=intrinsics.fy,
                    fy=intrinsics.fx,
                    ppx=intrinsics.height - 1 - intrinsics.ppy,
                    ppy=intrinsics.ppx,
                ),
            )
        if rotation_deg == 180:
            return (
                np.rot90(color_bgr, k=2).copy(),
                np.rot90(depth_mm, k=2).copy(),
                CameraIntrinsics(
                    width=intrinsics.width,
                    height=intrinsics.height,
                    fx=intrinsics.fx,
                    fy=intrinsics.fy,
                    ppx=intrinsics.width - 1 - intrinsics.ppx,
                    ppy=intrinsics.height - 1 - intrinsics.ppy,
                ),
            )
        if rotation_deg == 270:
            return (
                np.rot90(color_bgr, k=1).copy(),
                np.rot90(depth_mm, k=1).copy(),
                CameraIntrinsics(
                    width=intrinsics.height,
                    height=intrinsics.width,
                    fx=intrinsics.fy,
                    fy=intrinsics.fx,
                    ppx=intrinsics.ppy,
                    ppy=intrinsics.width - 1 - intrinsics.ppx,
                ),
            )
        raise ValueError("camera.image_rotation_deg must be one of 0, 90, 180, 270")

    def __enter__(self) -> "RealSenseRGBDCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
