"""RGB-D frame adapter for the running ``sensor_manager`` ROS graph."""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from .realsense_camera import CameraIntrinsics, RGBDFrame, RealSenseRGBDCamera


class SensorManagerRGBDCamera:
    """Consume compressed RGB-D and intrinsics without opening the D435 device.

    The sensor manager owns the physical D435.  This adapter only subscribes
    to its three published streams and therefore can run alongside the normal
    robot perception stack.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._node: Any | None = None
        self._rclpy: Any | None = None
        self._context: Any | None = None
        self._executor: Any | None = None
        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._color_t: float | None = None
        self._depth_t: float | None = None

    def start(self) -> None:
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CameraInfo, CompressedImage
        except ImportError as exc:
            raise RuntimeError("ROS sensor_msgs are required for sensor_manager RGB-D") from exc
        self._context = Context()
        rclpy.init(context=self._context)
        self._rclpy = rclpy
        self._node = rclpy.create_node("z1_sensor_manager_rgbd", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=2)
        self._node.create_subscription(CompressedImage, str(self.config.get("color_topic", "/sensor/rgbd_image")),
                                       self._on_color, qos)
        self._node.create_subscription(CompressedImage, str(self.config.get("depth_topic", "/sensor/rgbd_depth_image")),
                                       self._on_depth, qos)
        self._node.create_subscription(CameraInfo, str(self.config.get("camera_info_topic", "/sensor/rgbd_camera_info")),
                                       self._on_info, qos)

    def _on_color(self, message: Any) -> None:
        encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is not None:
            with self._lock:
                self._color, self._color_t = image, time.monotonic()

    def _on_depth(self, message: Any) -> None:
        encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is not None and image.ndim == 2 and image.dtype == np.uint16:
            with self._lock:
                self._depth, self._depth_t = image, time.monotonic()

    def _on_info(self, message: Any) -> None:
        if len(message.k) < 9 or message.width <= 0 or message.height <= 0:
            return
        intrinsics = CameraIntrinsics(width=int(message.width), height=int(message.height),
                                      fx=float(message.k[0]), fy=float(message.k[4]),
                                      ppx=float(message.k[2]), ppy=float(message.k[5]))
        with self._lock:
            self._intrinsics = intrinsics

    def capture(self, *, after_monotonic_s: float | None = None) -> RGBDFrame:
        """Return synchronized RGB-D, optionally requiring a frame newer than ``after_monotonic_s``."""
        if self._node is None or self._rclpy is None:
            raise RuntimeError("sensor_manager camera is not started")
        timeout_s = float(self.config.get("frame_timeout_ms", 5000)) / 1000.0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            assert self._executor is not None
            self._executor.spin_once(timeout_sec=min(0.1, deadline - time.monotonic()))
            with self._lock:
                color, depth, intrinsics = self._color, self._depth, self._intrinsics
                color_t, depth_t = self._color_t, self._depth_t
            if color is None or depth is None or intrinsics is None or color_t is None or depth_t is None:
                continue
            if after_monotonic_s is not None and (color_t <= after_monotonic_s or depth_t <= after_monotonic_s):
                continue
            if abs(color_t - depth_t) > 0.15:
                continue
            if color.shape[:2] != depth.shape:
                raise RuntimeError(f"sensor_manager RGB-D dimensions differ: {color.shape} vs {depth.shape}")
            if (intrinsics.width, intrinsics.height) != (color.shape[1], color.shape[0]):
                raise RuntimeError("sensor_manager CameraInfo does not match RGB-D dimensions")
            color, depth, intrinsics = RealSenseRGBDCamera._rotate_aligned_frame(
                color.copy(), depth.copy(), intrinsics, int(self.config.get("image_rotation_deg", 0)),
            )
            return RGBDFrame(color_bgr=color, depth_mm=depth, intrinsics=intrinsics,
                             depth_scale_m=0.001, captured_monotonic_s=time.monotonic())
        raise RuntimeError("timed out waiting for synchronized sensor_manager RGB-D and CameraInfo")

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._rclpy is not None and self._context is not None:
            self._rclpy.shutdown(context=self._context)
        self._rclpy = None
        self._context = None

    def __enter__(self) -> "SensorManagerRGBDCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
