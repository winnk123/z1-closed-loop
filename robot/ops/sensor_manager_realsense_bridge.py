#!/usr/bin/env python3
"""Publish the chest D435 as sensor_manager-compatible compressed RGB-D ROS topics."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from z1_planning.realsense_camera import RealSenseRGBDCamera


def main() -> int:
    import rclpy
    from sensor_msgs.msg import CameraInfo, CompressedImage

    rclpy.init()
    node = rclpy.create_node("sensor_rgbd")
    color_pub = node.create_publisher(CompressedImage, "/sensor/rgbd_image", 2)
    depth_pub = node.create_publisher(CompressedImage, "/sensor/rgbd_depth_image", 2)
    info_pub = node.create_publisher(CameraInfo, "/sensor/rgbd_camera_info", 2)
    camera_config = {
        "serial": os.environ.get("Z1_CAMERA_SERIAL", ""),
        "color_width": 1280, "color_height": 720,
        "depth_width": 640, "depth_height": 480,
        "fps": 15, "warmup_frames": 30, "frame_timeout_ms": 5000,
        "image_rotation_deg": 0,
    }
    try:
        while rclpy.ok():
            try:
                # A USB reconnect invalidates a RealSense pipeline permanently.
                # Recreate it instead of allowing sensor_manager to remain
                # "active" while its RGB-D bridge has silently exited.
                with RealSenseRGBDCamera(camera_config) as camera:
                    while rclpy.ok():
                        rclpy.spin_once(node, timeout_sec=0.0)
                        frame = camera.capture()
                        ok_color, color_data = cv2.imencode(
                            ".jpg", frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90],
                        )
                        ok_depth, depth_data = cv2.imencode(".png", frame.depth_mm)
                        if not ok_color or not ok_depth:
                            node.get_logger().error("failed to encode RGB-D frame")
                            continue
                        stamp = node.get_clock().now().to_msg()
                        color = CompressedImage()
                        color.header.stamp = stamp
                        color.header.frame_id = "sensor_rgbd_link"
                        color.format = "jpeg"
                        color.data = color_data.tobytes()
                        depth = CompressedImage()
                        depth.header.stamp = stamp
                        depth.header.frame_id = "sensor_rgbd_link"
                        depth.format = "png; 16UC1"
                        depth.data = depth_data.tobytes()
                        info = CameraInfo()
                        info.header.stamp = stamp
                        info.header.frame_id = "sensor_rgbd_link"
                        info.width, info.height = frame.intrinsics.width, frame.intrinsics.height
                        info.k = [frame.intrinsics.fx, 0.0, frame.intrinsics.ppx,
                                  0.0, frame.intrinsics.fy, frame.intrinsics.ppy,
                                  0.0, 0.0, 1.0]
                        info.p = [frame.intrinsics.fx, 0.0, frame.intrinsics.ppx, 0.0,
                                  0.0, frame.intrinsics.fy, frame.intrinsics.ppy, 0.0,
                                  0.0, 0.0, 1.0, 0.0]
                        color_pub.publish(color)
                        depth_pub.publish(depth)
                        info_pub.publish(info)
            except RuntimeError as exc:
                if rclpy.ok():
                    node.get_logger().error(f"D435 bridge reconnecting after: {exc}")
                    time.sleep(2.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
