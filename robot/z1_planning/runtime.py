"""Hardware adapters used by the Z1 closed-loop command-line runner.

ROS and the Z1 SDK are imported only when an adapter is instantiated.  This
keeps path-following and safety modules importable on development machines.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .gyro_heading import GyroHeadingState, GyroHeadingTracker
from .path_follower import OdomPose
from .safety_bridge import Z1RuntimeState


@dataclass(frozen=True)
class LidarSlamHealthConfig:
    """Local-control admission policy for ``eamegrapher_3d`` output."""

    min_local_match_score: float = 0.40
    state_timeout_s: float = 0.20
    max_pose_jump_m: float = 0.20
    require_localization_succeed: bool = False

    def __post_init__(self) -> None:
        for name in ("min_local_match_score", "state_timeout_s", "max_pose_jump_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LidarSlamStatus:
    """Subset of ``eamegrapher_msgs/msg/SlamStates`` used by the gate."""

    timestamp_s: float
    mode: int
    localization_succeed: bool
    local_match_score: float
    global_motion_score: float
    local_motion_score: float
    pose_jump_detected: bool


def lidar_slam_is_healthy(
    status: LidarSlamStatus | None,
    *,
    now_s: float,
    config: LidarSlamHealthConfig,
) -> bool:
    """Return whether a fresh LiDAR-SLAM status admits local path tracking."""
    if status is None or status.pose_jump_detected:
        return False
    values = (now_s, status.timestamp_s, status.local_match_score)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    age = float(now_s) - status.timestamp_s
    if age < 0.0 or age > config.state_timeout_s:
        return False
    if status.local_match_score < config.min_local_match_score:
        return False
    return not config.require_localization_succeed or status.localization_succeed


def quaternion_yaw(quaternion: Any) -> float:
    """Return a planar yaw angle from a ROS-style quaternion object."""
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def transform_pose(message: Any) -> OdomPose:
    """Extract the child pose expressed in the parent frame from a TF message."""
    transform = message.transform
    return OdomPose(
        float(transform.translation.x),
        float(transform.translation.y),
        quaternion_yaw(transform.rotation),
    )


def transform_stamp_ns(message: Any) -> int:
    """Return the ROS timestamp carried by a TransformStamped message."""
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_timestamp_s(message: Any) -> float | None:
    """Read a valid ROS header timestamp without mixing it with receipt time."""
    try:
        stamp = message.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    except (AttributeError, TypeError, ValueError):
        return None
    return timestamp_s if math.isfinite(timestamp_s) and timestamp_s > 0.0 else None


def relative_pose(current: OdomPose, origin: OdomPose) -> OdomPose:
    """Express an odometry pose in the initial body frame."""
    dx, dy = current.x_m - origin.x_m, current.y_m - origin.y_m
    cosine, sine = math.cos(origin.yaw_rad), math.sin(origin.yaw_rad)
    yaw = math.atan2(
        math.sin(current.yaw_rad - origin.yaw_rad),
        math.cos(current.yaw_rad - origin.yaw_rad),
    )
    return OdomPose(cosine * dx + sine * dy, -sine * dx + cosine * dy, yaw)


class Z1RosState:
    """Latest ROS state plus raw telemetry retained for the execution audit."""

    def __init__(
        self,
        *,
        pose_source: str = "chassis",
        lidar_slam_config: LidarSlamHealthConfig | None = None,
    ) -> None:
        import rclpy
        from motion_msgs.msg import State
        from nav_msgs.msg import Odometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool

        if pose_source not in {"chassis", "lidar_slam", "lidar_local"}:
            raise ValueError("pose_source must be 'chassis', 'lidar_slam', or 'lidar_local'")
        self.rclpy = rclpy
        self.pose_source = pose_source
        self.pose_topic = {
            "chassis": "/odom_chassis",
            "lidar_slam": "/odom (map -> pelvis)",
            "lidar_local": "TF odom -> pelvis",
        }[pose_source]
        self.lidar_slam_config = lidar_slam_config or LidarSlamHealthConfig()
        rclpy.init()
        self.node = rclpy.create_node("z1_closed_loop")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.odom: OdomPose | None = None
        self.origin: OdomPose | None = None
        self.robot_fsm: int | None = None
        self.robot_hanged: bool | None = None
        self.estop: bool | None = None
        self.odom_t: float | None = None
        self.imu_t: float | None = None
        self.imu_measurement_t: float | None = None
        self.imu_yaw_rad: float | None = None
        self.imu_angular_velocity_z_rps: float | None = None
        self.task_heading: GyroHeadingState | None = None
        self._task_heading_tracker: GyroHeadingTracker | None = None
        self.odom_samples: list[dict[str, Any]] = []
        self.lidar_slam_odom_samples: list[dict[str, Any]] = []
        self.lidar_slam_state_samples: list[dict[str, Any]] = []
        self._lidar_last_pose: OdomPose | None = None
        self._lidar_pose_jump_detected = False
        self._lidar_slam_status: LidarSlamStatus | None = None
        self._lidar_local_last_stamp_ns: int | None = None
        self.lidar_local_tf_samples: list[dict[str, Any]] = []
        self._tf_buffer: Any | None = None
        self.imu_samples: list[dict[str, Any]] = []
        self.leg_state_samples: list[dict[str, Any]] = []
        self.hanged_samples: list[dict[str, Any]] = []
        self.estop_samples: list[dict[str, Any]] = []
        self.node.create_subscription(Odometry, "/odom_chassis", self._chassis_odom, qos)
        if self.pose_source in {"lidar_slam", "lidar_local"}:
            from eamegrapher_msgs.msg import SlamStates

            if self.pose_source == "lidar_slam":
                self.node.create_subscription(Odometry, "/odom", self._lidar_slam_odom, qos)
            self.node.create_subscription(SlamStates, "/slam_states", self._lidar_slam_state, qos)
        if self.pose_source == "lidar_local":
            from tf2_ros import Buffer, TransformListener

            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self.node, spin_thread=False)
        self.node.create_subscription(Imu, "/imu", self._imu, qos)
        self.node.create_subscription(State, "/leg_state", self._leg_state, qos)
        self.node.create_subscription(Bool, "/manager/robot_hanged", self._hanged, qos)
        self.node.create_subscription(Bool, "/emy_stop", self._estop, qos)

    @staticmethod
    def _odom_pose(message: Any) -> OdomPose:
        return OdomPose(
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            quaternion_yaw(message.pose.pose.orientation),
        )

    @staticmethod
    def _odom_sample(message: Any, pose: OdomPose, timestamp_s: float) -> dict[str, Any]:
        return {
            "t_s": timestamp_s,
            "frame_id": str(message.header.frame_id),
            "child_frame_id": str(message.child_frame_id),
            "pose": {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad},
            "linear_velocity_mps": {
                "x": float(message.twist.twist.linear.x),
                "y": float(message.twist.twist.linear.y),
                "z": float(message.twist.twist.linear.z),
            },
            "angular_velocity_rps": {
                "x": float(message.twist.twist.angular.x),
                "y": float(message.twist.twist.angular.y),
                "z": float(message.twist.twist.angular.z),
            },
        }

    def _select_odom(self, pose: OdomPose, timestamp_s: float, source: str) -> None:
        if self.pose_source != source:
            return
        self.odom = pose
        self.odom_t = timestamp_s
        if self.origin is None:
            self.origin = pose

    def _chassis_odom(self, message: Any) -> None:
        timestamp_s = time.monotonic()
        pose = self._odom_pose(message)
        self.odom_samples.append(self._odom_sample(message, pose, timestamp_s))
        self._select_odom(pose, timestamp_s, "chassis")

    def _lidar_slam_odom(self, message: Any) -> None:
        timestamp_s = time.monotonic()
        pose = self._odom_pose(message)
        if self._lidar_last_pose is not None:
            jump = math.hypot(pose.x_m - self._lidar_last_pose.x_m, pose.y_m - self._lidar_last_pose.y_m)
            if jump > self.lidar_slam_config.max_pose_jump_m:
                self._lidar_pose_jump_detected = True
        self._lidar_last_pose = pose
        self.lidar_slam_odom_samples.append(self._odom_sample(message, pose, timestamp_s))
        self._select_odom(pose, timestamp_s, "lidar_slam")

    def _lidar_slam_state(self, message: Any) -> None:
        timestamp_s = time.monotonic()
        self._lidar_slam_status = LidarSlamStatus(
            timestamp_s=timestamp_s,
            mode=int(message.mode),
            localization_succeed=bool(message.localization_succeed),
            local_match_score=float(message.local_match_score),
            global_motion_score=float(message.global_motion_score),
            local_motion_score=float(message.local_motion_score),
            pose_jump_detected=self._lidar_pose_jump_detected,
        )
        self.lidar_slam_state_samples.append({
            "t_s": timestamp_s,
            "mode": self._lidar_slam_status.mode,
            "map_name": str(message.map_name),
            "localization_succeed": self._lidar_slam_status.localization_succeed,
            "local_match_score": self._lidar_slam_status.local_match_score,
            "global_motion_score": self._lidar_slam_status.global_motion_score,
            "local_motion_score": self._lidar_slam_status.local_motion_score,
            "pose_jump_detected": self._lidar_slam_status.pose_jump_detected,
        })

    def _lidar_local_tf(self) -> None:
        """Read the continuous local pose from TF without consuming map corrections."""
        if self._tf_buffer is None:
            return
        from rclpy.time import Time
        from tf2_ros import TransformException

        try:
            message = self._tf_buffer.lookup_transform("odom", "pelvis", Time())
        except TransformException:
            return
        stamp_ns = transform_stamp_ns(message)
        # lookup_transform returns the latest cached transform repeatedly.  Do
        # not make stale TF look fresh to the command watchdog.
        if stamp_ns <= 0 or stamp_ns == self._lidar_local_last_stamp_ns:
            return
        pose = transform_pose(message)
        timestamp_s = time.monotonic()
        if self._lidar_last_pose is not None:
            jump = math.hypot(pose.x_m - self._lidar_last_pose.x_m, pose.y_m - self._lidar_last_pose.y_m)
            if jump > self.lidar_slam_config.max_pose_jump_m:
                self._lidar_pose_jump_detected = True
        self._lidar_last_pose = pose
        self._lidar_local_last_stamp_ns = stamp_ns
        self.lidar_local_tf_samples.append({
            "t_s": timestamp_s,
            "tf_stamp_ns": stamp_ns,
            "parent_frame": str(message.header.frame_id),
            "child_frame": str(message.child_frame_id),
            "pose": {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad},
        })
        self._select_odom(pose, timestamp_s, "lidar_local")

    def _imu(self, message: Any) -> None:
        self.imu_t = time.monotonic()
        self.imu_measurement_t = message_timestamp_s(message) or self.imu_t
        self.imu_yaw_rad = quaternion_yaw(message.orientation)
        self.imu_angular_velocity_z_rps = float(message.angular_velocity.z)
        if self._task_heading_tracker is not None:
            self.task_heading = self._task_heading_tracker.update(
                gyro_timestamp_s=self.imu_measurement_t,
                gyro_z_rps=self.imu_angular_velocity_z_rps,
            )
        self.imu_samples.append({
            "t_s": self.imu_t,
            "measurement_t_s": self.imu_measurement_t,
            "yaw_rad": self.imu_yaw_rad,
            "orientation": {
                "x": float(message.orientation.x), "y": float(message.orientation.y),
                "z": float(message.orientation.z), "w": float(message.orientation.w),
            },
            "angular_velocity_rps": {
                "x": float(message.angular_velocity.x), "y": float(message.angular_velocity.y),
                "z": float(message.angular_velocity.z),
            },
            "linear_acceleration_mps2": {
                "x": float(message.linear_acceleration.x), "y": float(message.linear_acceleration.y),
                "z": float(message.linear_acceleration.z),
            },
        })

    def begin_task_heading(self, *, gyro_bias_rps: float, max_sample_gap_s: float = 0.10) -> GyroHeadingState:
        """Define the current physical orientation as task heading zero.

        This intentionally uses the next/fresh IMU stream only.  It does not
        consume the unstable quaternion yaw or any LiDAR-SLAM pose estimate.
        """
        if self.imu_measurement_t is None or self.imu_angular_velocity_z_rps is None:
            raise RuntimeError("cannot start task heading before a fresh IMU sample")
        tracker = GyroHeadingTracker(
            gyro_bias_rps=gyro_bias_rps,
            max_sample_gap_s=max_sample_gap_s,
        )
        self.task_heading = tracker.start(
            gyro_timestamp_s=self.imu_measurement_t,
            gyro_z_rps=self.imu_angular_velocity_z_rps,
        )
        self._task_heading_tracker = tracker
        return self.task_heading

    def _leg_state(self, message: Any) -> None:
        self.robot_fsm = int(message.robot_fsm)
        self.leg_state_samples.append({
            "t_s": time.monotonic(), "robot_fsm": self.robot_fsm,
            "switch_flag": int(message.switch_flag),
        })

    def _hanged(self, message: Any) -> None:
        self.robot_hanged = bool(message.data)
        self.hanged_samples.append({"t_s": time.monotonic(), "value": self.robot_hanged})

    def _estop(self, message: Any) -> None:
        self.estop = bool(message.data)
        self.estop_samples.append({"t_s": time.monotonic(), "value": self.estop})

    def spin(self, *, wait_timeout_s: float = 0.005) -> None:
        """Receive one ROS callback with a bounded wait.

        The Fast DDS endpoint on Z1 starves the IMU callback when a single
        executor is repeatedly spun with zero timeout to drain a queue.  One
        bounded wait reliably receives the next source-timestamped IMU frame;
        callers invoke this method at their desired control rate.
        """
        self.rclpy.spin_once(self.node, timeout_sec=wait_timeout_s)
        if self.pose_source == "lidar_local":
            self._lidar_local_tf()

    def snapshot(self) -> Z1RuntimeState:
        if self.pose_source not in {"lidar_slam", "lidar_local"}:
            return Z1RuntimeState(self.robot_hanged, self.estop, self.robot_fsm, self.odom_t, self.imu_t)
        status = self._lidar_slam_status
        return Z1RuntimeState(
            self.robot_hanged,
            self.estop,
            self.robot_fsm,
            self.odom_t,
            self.imu_t,
            localization_healthy=lidar_slam_is_healthy(
                status, now_s=time.monotonic(), config=self.lidar_slam_config,
            ),
            localization_timestamp_s=status.timestamp_s if status is not None else None,
        )

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


class Z1SdkSender:
    """Narrow adapter around the official high-level joystick interface."""

    def __init__(self, local_ip: str) -> None:
        sdk_root = os.environ.get("Z1_SDK_ROOT", "")
        if not sdk_root:
            raise RuntimeError("Z1_SDK_ROOT is required")
        sdk_lib = Path(sdk_root) / "lib" / platform.machine()
        if sdk_lib.is_dir():
            sys.path.insert(0, str(sdk_lib))
            os.environ["LD_LIBRARY_PATH"] = str(sdk_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        import magicbot_z1_python as magicbot
        from .gait import _initialize_high_level_controller

        self.sdk, self.robot = magicbot, magicbot.MagicRobot()
        self.robot.initialize(local_ip)
        status = self.robot.connect()
        if hasattr(status, "code") and int(status.code) != 0:
            raise RuntimeError(f"MagicRobot.connect failed: {status}")
        self.controller = _initialize_high_level_controller(self.robot, magicbot)

    def send(self, axes: Any) -> None:
        command = self.sdk.JoystickCommand()
        command.left_x_axis = axes.left_x_axis
        command.left_y_axis = axes.left_y_axis
        command.right_x_axis = axes.right_x_axis
        command.right_y_axis = axes.right_y_axis
        status = self.controller.send_joystick_command(command)
        if hasattr(status, "code") and int(status.code) != 0:
            raise RuntimeError(f"send_joystick_command failed: {status}")

    def close(self) -> None:
        try:
            from .gait import _joystick_command, _send_joystick

            zero = _joystick_command(self.sdk)
            for _ in range(5):
                _send_joystick(self.controller, zero, self.sdk)
                time.sleep(0.05)
        finally:
            self.controller.shutdown()
            self.robot.disconnect()
            self.robot.shutdown()
