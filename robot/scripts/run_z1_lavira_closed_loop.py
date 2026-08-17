#!/usr/bin/env python3
"""Fail-closed Baseline-style LA -> VA -> NavDP -> Z1 task execution.

Each task step captures a front/right/left panorama, asks LA to update its
TODO checklist and select a heading, then uses one fresh RGB-D frame for VA.
The first NavDP path is truncated once by G1's safety margin; rolling NavDP
replans retain that truncated endpoint and never call VA again. The loop
repeats until LA or VA reports a final stop. No motion is possible unless both
execution flags and the exact confirmation are supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from z1_planning.config import load_config
from z1_planning.audit_json import json_safe
from z1_planning.geometry import LocalGoal
from z1_planning.heading_correction import correction_from_settled_yaw
from z1_planning.path_follower import OdomPose, PathFollower, PathFollowerConfig
from z1_planning.pipeline import Z1PlanningPipeline
from z1_planning.realsense_camera import RGBDFrame, RealSenseRGBDCamera
from z1_planning.relay_client import RelayClient
from z1_planning.runtime import LidarSlamHealthConfig, Z1RosState, Z1SdkSender
from z1_planning.safety_bridge import Z1JoystickAxes, Z1SafetyBridge, Z1SafetyBridgeConfig
from z1_planning.sensor_manager_camera import SensorManagerRGBDCamera
from z1_planning.strategic_navigation import HeadingTurnConfig, HeadingTurnController, parse_strategic_decision, wrap_angle
from z1_planning.visual_loop import (
    BaselineTravelPolicy, VisualLoopConfig, admit_baseline_replan, baseline_goal_reached, constrain_replan_to_fixed_goal, odom_goal_to_local,
    reserve_baseline_execution_goal,
)


CONFIRMATION = "EXECUTE_Z1_LAVIRA_CLOSED_LOOP"
VIEW_TURNS = (("right", "right"), ("left", "left"))
SCAN_STEP_DEGREES = (30, 60, 90)
CAMERA_SETTLE_S = 0.35
TURN_SETTLE_S = 0.50
TURN_SETTLE_SPREAD_RAD = math.radians(2.0)
# Require scans and LA-selected headings to settle within 5 degrees before
# accepting them; one bounded correction attempt remains available.
TURN_CONTROL_TOLERANCE_RAD = math.radians(5.0)
TURN_MAX_YAW_STEP_RAD = math.radians(12.0)
TURN_MAX_SETTLE_CORRECTIONS = 1
# The observed odom -> pelvis stream can have a short 0.24 s inter-sample
# gap. Keep the temporary mission watchdog above that jitter while retaining
# the stricter IMU and SLAM-health watchdogs.
MISSION_ODOM_TIMEOUT_S = 0.50
MISSION_LOCALIZATION_TIMEOUT_S = 0.40
LA_SELECTABLE_VIEWS = (
    "front_000deg",
    "right_030deg", "right_060deg", "right_090deg",
    "left_030deg", "left_060deg", "left_090deg",
)


@dataclass(frozen=True)
class FrameJob:
    frame: RGBDFrame
    origin: OdomPose
    artifact_prefix: str = "planning"
    va_override: dict[str, Any] | None = None
    allow_va_stop_for_final_approach: bool = False
    final_stop_requested: bool = False
    fixed_odom_goal: tuple[float, float] | None = None


@dataclass(frozen=True)
class PlanResult:
    job: FrameJob
    planning: dict[str, Any] | None = None
    error: str | None = None
    started_s: float = 0.0
    completed_s: float = 0.0


class MissionVideoRecorder:
    """Record sensor_manager RGB and visualized depth for one mission."""

    def __init__(self, output_dir: Path, fps: float) -> None:
        self.output_dir = output_dir
        self.fps = float(fps)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._color_writer: cv2.VideoWriter | None = None
        self._depth_writer: cv2.VideoWriter | None = None
        self._last_write_s = float("-inf")
        self.frame_count = 0
        self.error: str | None = None

    def write(self, frame: RGBDFrame) -> None:
        if self.error or frame.captured_monotonic_s - self._last_write_s < 1.0 / self.fps:
            return
        height, width = frame.color_bgr.shape[:2]
        if self._color_writer is None:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._color_writer = cv2.VideoWriter(str(self.output_dir / "rgb.mp4"), codec, self.fps, (width, height))
            self._depth_writer = cv2.VideoWriter(str(self.output_dir / "depth_visual.mp4"), codec, self.fps, (width, height))
            if not self._color_writer.isOpened() or not self._depth_writer.isOpened():
                self.error = "failed to open RGB-D video writers"
                return
        depth_visual = cv2.applyColorMap(
            cv2.convertScaleAbs(frame.depth_mm, alpha=255.0 / 4000.0), cv2.COLORMAP_TURBO,
        )
        self._color_writer.write(frame.color_bgr)
        self._depth_writer.write(depth_visual)
        self._last_write_s = frame.captured_monotonic_s
        self.frame_count += 1

    def close(self) -> dict[str, Any]:
        for writer in (self._color_writer, self._depth_writer):
            if writer is not None:
                writer.release()
        return {
            "directory": str(self.output_dir), "rgb_video": str(self.output_dir / "rgb.mp4"),
            "depth_visual_video": str(self.output_dir / "depth_visual.mp4"),
            "fps": self.fps, "frame_count": self.frame_count, "error": self.error,
        }


class CameraWorker:
    """Use sensor_manager by default, so the runner never owns the D435."""

    def __init__(self, config: dict[str, Any], source: str, recorder: MissionVideoRecorder | None = None) -> None:
        self._config, self._source = config, source
        self._recorder = recorder
        self._lock = threading.Lock()
        self._frame: RGBDFrame | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="z1-lavira-camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6.0)

    def latest(self) -> tuple[RGBDFrame | None, str | None]:
        with self._lock:
            return self._frame, self._error

    def _run(self) -> None:
        try:
            camera_class = SensorManagerRGBDCamera if self._source == "sensor_manager" else RealSenseRGBDCamera
            with camera_class(self._config) as camera:
                while not self._stop.is_set():
                    frame = camera.capture()
                    if self._recorder is not None:
                        self._recorder.write(frame)
                    with self._lock:
                        self._frame = frame
        except Exception as exc:
            with self._lock:
                self._error = str(exc)


class PlannerWorker:
    """Keep the relay session started by LA while NavDP replans asynchronously."""

    def __init__(self, config: dict[str, Any], relay: RelayClient) -> None:
        self._config, self._relay = config, relay
        self.jobs: queue.Queue[FrameJob] = queue.Queue(maxsize=1)
        self.results: queue.Queue[PlanResult] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="z1-lavira-navdp", daemon=True)
        self._busy, self.start_error = False, None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6.0)

    def submit(self, job: FrameJob) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        try:
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            with self._lock:
                self._busy = False
            return False

    def _run(self) -> None:
        try:
            pipeline = Z1PlanningPipeline(self._config, self._relay)
            # LA owns sequence 0; do not reset the same relay session before VA.
            pipeline.adopt_session()
        except Exception as exc:
            self.start_error = str(exc)
            return
        while not self._stop.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                if job.fixed_odom_goal is None:
                    planning = pipeline.run(
                        job.frame,
                        reset_session=False,
                        va_override=job.va_override,
                        allow_va_stop_for_final_approach=job.allow_va_stop_for_final_approach,
                        final_stop_requested=job.final_stop_requested,
                        artifact_prefix=job.artifact_prefix,
                    )
                    if planning.get("va_requested_stop") and not job.allow_va_stop_for_final_approach:
                        result = PlanResult(job, planning=planning, started_s=started, completed_s=time.monotonic())
                        self.results.put(result)
                        continue
                    if planning.get("final_stop_no_bbox"):
                        result = PlanResult(job, planning=planning, started_s=started, completed_s=time.monotonic())
                        self.results.put(result)
                        continue
                else:
                    forward, left = odom_goal_to_local(*job.fixed_odom_goal, job.origin)
                    planning = pipeline.run(
                        job.frame, reset_session=False,
                        local_goal_override=LocalGoal(
                            pixel_u=-1, pixel_v=-1, depth_m=float(math.hypot(forward, left)),
                            x_forward_m=forward, y_left_m=left,
                        ),
                        write_artifacts=False,
                    )
                if job.fixed_odom_goal is not None:
                    planning["odom_goal"] = {
                        "x_m": job.fixed_odom_goal[0], "y_m": job.fixed_odom_goal[1],
                    }
                result = PlanResult(job, planning=planning, started_s=started, completed_s=time.monotonic())
            except Exception as exc:
                result = PlanResult(job, error=str(exc), started_s=started, completed_s=time.monotonic())
            finally:
                with self._lock:
                    self._busy = False
            self.results.put(result)


def drain_results(worker: PlannerWorker) -> list[PlanResult]:
    result: list[PlanResult] = []
    while True:
        try:
            result.append(worker.results.get_nowait())
        except queue.Empty:
            return result


def wait_for_frame(camera: CameraWorker, after_s: float, timeout_s: float = 5.0) -> RGBDFrame:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame, error = camera.latest()
        if error:
            raise RuntimeError("camera_failed:" + error)
        if frame is not None and frame.captured_monotonic_s > after_s:
            return frame
        time.sleep(0.02)
    raise RuntimeError("camera_frame_timeout")


def preflight(state: Z1RosState, timeout_s: float = 6.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state.spin()
        snapshot = state.snapshot()
        now = time.monotonic()
        odom_age = now - state.odom_t if state.odom_t is not None else math.inf
        imu_age = now - state.imu_t if state.imu_t is not None else math.inf
        localization_ok = state.pose_source != "lidar_local" or snapshot.localization_healthy is True
        if (state.odom is not None and snapshot.robot_hanged is False and snapshot.estop is False
                and snapshot.robot_fsm == 46 and localization_ok
                and 0.0 <= odom_age <= MISSION_ODOM_TIMEOUT_S and 0.0 <= imu_age <= 0.20):
            return
        time.sleep(0.02)
    raise RuntimeError("motion_preflight_rejected:" + repr(state.snapshot()))


def wait_for_fresh_turn_inputs(state: Z1RosState, bridge: Z1SafetyBridge, timeout_s: float = 0.18) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state.spin(wait_timeout_s=0.02)
        now = time.monotonic()
        runtime = state.snapshot()
        odom_age = now - runtime.odom_timestamp_s if runtime.odom_timestamp_s is not None else math.inf
        imu_age = now - runtime.imu_timestamp_s if runtime.imu_timestamp_s is not None else math.inf
        if 0.0 <= odom_age <= bridge.config.odom_timeout_s and 0.0 <= imu_age <= bridge.config.imu_timeout_s:
            return
    raise RuntimeError("turn_control_inputs_stale")


def wait_for_settled_yaw(state: Z1RosState, timeout_s: float = 4.0) -> float:
    """Return a recent yaw only after its samples remain within the settle spread."""
    deadline = time.monotonic() + timeout_s
    samples: list[tuple[float, float]] = []
    last_t: float | None = None
    while time.monotonic() < deadline:
        state.spin(wait_timeout_s=0.05)
        if state.odom is None or state.odom_t is None or state.odom_t == last_t:
            continue
        samples.append((state.odom_t, state.odom.yaw_rad))
        last_t = state.odom_t
        window = [(t, yaw) for t, yaw in samples if t >= samples[-1][0] - TURN_SETTLE_S - 0.05]
        if len(window) < 4 or window[-1][0] - window[0][0] < TURN_SETTLE_S:
            continue
        reference = window[0][1]
        spread = max(abs(wrap_angle(yaw - reference)) for _, yaw in window)
        if spread <= TURN_SETTLE_SPREAD_RAD:
            return window[-1][1]
    raise RuntimeError("turn_yaw_not_settled")


def drive_to_yaw(
    state: Z1RosState, bridge: Z1SafetyBridge, sender: Z1SdkSender, *, target_yaw_rad: float,
    phase: str, audit: list[dict[str, Any]], config: HeadingTurnConfig,
) -> float:
    """Command one absolute odom-yaw target, rejecting discontinuous odometry."""
    preflight(state, timeout_s=2.0)
    wait_for_fresh_turn_inputs(state, bridge)
    if state.odom is None:
        raise RuntimeError(phase + ":odom_missing")
    controller = HeadingTurnController(state.odom.yaw_rad, "front", time.monotonic(), config)
    controller.target_yaw_rad = wrap_angle(target_yaw_rad)
    previous_yaw = state.odom.yaw_rad
    try:
        while True:
            wait_for_fresh_turn_inputs(state, bridge)
            now = time.monotonic()
            runtime = state.snapshot()
            if state.odom is None:
                desired_yaw, complete, failure, yaw_error = 0.0, False, "odom_missing", None
            else:
                yaw_step = abs(wrap_angle(state.odom.yaw_rad - previous_yaw))
                previous_yaw = state.odom.yaw_rad
                if yaw_step > TURN_MAX_YAW_STEP_RAD:
                    desired_yaw, complete, failure = 0.0, False, "odom_yaw_jump"
                    yaw_error = wrap_angle(target_yaw_rad - state.odom.yaw_rad)
                else:
                    command = controller.update(state.odom.yaw_rad, now)
                    desired_yaw, complete, failure, yaw_error = (
                        command.desired_yaw_rate_radps, command.complete,
                        command.failure_reason, command.yaw_error_rad,
                    )
            output = bridge.evaluate(
                desired_vx_mps=0.0, desired_vyaw_rps=desired_yaw, now_s=now,
                command_timestamp_s=now, runtime_state=runtime,
                motion_enabled=not complete and failure is None,
            )
            sender.send(output.axes)
            audit.append({
                "phase": phase, "t_s": now,
                "current_yaw_rad": state.odom.yaw_rad if state.odom else None,
                "target_yaw_rad": target_yaw_rad, "yaw_error_rad": yaw_error,
                "bridge": asdict(output),
            })
            if complete:
                return state.odom.yaw_rad if state.odom else math.nan
            if failure is not None or not output.active:
                raise RuntimeError(phase + ":" + (failure or output.stop_reason))
            time.sleep(0.05)
    finally:
        sender.send(Z1JoystickAxes.zero())


def drive_and_settle_to_yaw(
    state: Z1RosState, bridge: Z1SafetyBridge, sender: Z1SdkSender, *, target_yaw_rad: float,
    phase: str, audit: list[dict[str, Any]], config: HeadingTurnConfig,
) -> tuple[float, float, float]:
    """Reach an absolute target and correct the residual observed after the body settles."""
    final_yaw = settled_yaw = residual = math.nan
    for attempt in range(TURN_MAX_SETTLE_CORRECTIONS + 1):
        final_yaw = drive_to_yaw(
            state, bridge, sender, target_yaw_rad=target_yaw_rad,
            phase=f"{phase}_attempt_{attempt + 1}", audit=audit, config=config,
        )
        settled_yaw = wait_for_settled_yaw(state)
        correction = correction_from_settled_yaw(
            target_yaw_rad=target_yaw_rad, settled_yaw_rad=settled_yaw,
            tolerance_rad=TURN_CONTROL_TOLERANCE_RAD,
        )
        residual = correction.residual_rad
        audit.append({
            "phase": phase, "event": "post_stop_settle", "attempt": attempt + 1,
            "target_yaw_rad": target_yaw_rad, "settled_yaw_rad": settled_yaw,
            "residual_degrees": math.degrees(residual), "correction_direction": correction.direction,
            "correction_degrees": correction.degrees,
            "within_settled_tolerance": correction.within_tolerance,
        })
        if correction.within_tolerance:
            break
    return final_yaw, settled_yaw, residual


def save_frame(output_dir: Path, name: str, frame: RGBDFrame) -> dict[str, str]:
    color = output_dir / f"{name}_color.jpg"
    depth = output_dir / f"{name}_depth_mm.png"
    if not cv2.imwrite(str(color), frame.color_bgr) or not cv2.imwrite(str(depth), frame.depth_mm):
        raise RuntimeError("failed to write " + name + " RGB-D artifact")
    return {"color": str(color), "depth": str(depth)}


def annotate_tactical_view(
    output_dir: Path,
    *,
    task_step: int,
    pixel_bbox: list[int] | tuple[int, int, int, int],
) -> Path | None:
    """Draw VA's first valid bbox on its tactical input image.

    This only enriches an experiment artifact. A failed write is audited by
    the caller and must not affect robot control.
    """
    image_path = output_dir / f"step{task_step:03d}_tactical_color.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    left, top, right, bottom = (int(value) for value in pixel_bbox)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), thickness=3)
    return image_path if cv2.imwrite(str(image_path), image) else None


def labelled_observation(frame: RGBDFrame, view_label: str) -> np.ndarray:
    """Return a copy whose label identifies the exact camera pose LA may select."""
    image = frame.color_bgr.copy()
    cv2.rectangle(image, (0, 0), (360, 44), (0, 0, 0), thickness=-1)
    cv2.putText(image, view_label.upper(), (12, 31), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def wait_for_settled_frame(camera: CameraWorker, last_frame_s: float) -> RGBDFrame:
    """Discard turn-transition frames and return one captured after a zero-command settle period."""
    not_before_s = time.monotonic() + CAMERA_SETTLE_S
    while time.monotonic() < not_before_s:
        time.sleep(0.02)
    return wait_for_frame(camera, max(last_frame_s, not_before_s))


def scan_turn_steps(
    *, state: Z1RosState, bridge: Z1SafetyBridge, sender: Z1SdkSender,
    camera: CameraWorker, direction: str, label: str, angles_deg: tuple[int, ...],
    origin_yaw_rad: float, last_frame_s: float, output_dir: Path,
    audit: list[dict[str, Any]], turn_config: HeadingTurnConfig,
) -> tuple[dict[str, RGBDFrame], float]:
    """Visit absolute origin-relative views and save a fresh frame at each settled target."""
    frames: dict[str, RGBDFrame] = {}
    sign = 1.0 if direction == "left" else -1.0
    for angle_deg in angles_deg:
        phase = f"{label}_{angle_deg:03d}deg"
        target_yaw_rad = wrap_angle(origin_yaw_rad + sign * math.radians(angle_deg))
        drive_and_settle_to_yaw(
            state, bridge, sender, target_yaw_rad=target_yaw_rad,
            phase=phase, audit=audit, config=turn_config,
        )
        frame = wait_for_settled_frame(camera, last_frame_s)
        last_frame_s = frame.captured_monotonic_s
        frames[phase] = frame
        save_frame(output_dir, phase, frame)
    return frames, last_frame_s


def visual_history_info(task_steps: list[dict[str, Any]]) -> str:
    """Summarize completed Z1 task steps for the next LA decision."""
    if not task_steps:
        return "No visual history available yet."
    recent = task_steps[-3:]
    lines = []
    for step in recent:
        lines.append(
            "Step {step}: selected={selected}; landmark={landmark}; result={result}; "
            "travelled={travelled:.2f}m".format(
                step=step["step"],
                selected=step.get("selected_view") or "none",
                landmark=step.get("expected_landmark") or "unspecified",
                result=step.get("result") or "pending",
                travelled=float(step.get("travelled_m", 0.0)),
            )
        )
    return "\n".join(lines)


def execute_subgoal(
    *,
    state: Z1RosState,
    bridge: Z1SafetyBridge,
    sender: Z1SdkSender,
    camera: CameraWorker,
    relay: RelayClient,
    config: dict[str, Any],
    travel_policy: BaselineTravelPolicy,
    loop_config: VisualLoopConfig,
    replan_interval_s: float,
    mission_deadline_s: float,
    last_frame_s: float,
    task_step: int,
    run_dir: Path,
    planning_events: list[dict[str, Any]],
    control_audit: list[dict[str, Any]],
    terminate_after_approach: bool = False,
) -> tuple[str, float, float]:
    """Run one LA-selected landmark approach, returning a subgoal result.

    A local NavDP goal belongs to precisely one strategic subgoal.  A fresh
    worker for every task step prevents the previous landmark's fixed odom goal
    from being reused after LA advances its TODO list.  When LA has already
    requested STOP, its first VA result is interpreted as the Baseline final
    verification: a bbox permits one final approach; no bbox ends the task.
    """
    state.spin()
    if state.odom is None:
        raise RuntimeError("odom_missing_after_la_turn")
    tactical_frame = wait_for_settled_frame(camera, last_frame_s)
    last_frame_s = tactical_frame.captured_monotonic_s
    save_frame(Path(str(config["artifacts"]["directory"])), f"step{task_step:03d}_tactical", tactical_frame)
    planner = PlannerWorker(config, relay)
    follower: PathFollower | None = None
    last_pose: OdomPose | None = None
    travelled_m = 0.0
    failure_count = 0
    final_approach_requested = terminate_after_approach
    fixed_odom_goal: tuple[float, float] | None = None
    tactical_bbox_annotated = False
    try:
        planner.start()
        if not planner.submit(FrameJob(
            tactical_frame, state.odom,
            artifact_prefix=f"step{task_step:03d}_va",
            final_stop_requested=terminate_after_approach,
        )):
            return "initial_planner_submit_failed", last_frame_s, travelled_m
        while time.monotonic() < mission_deadline_s:
            state.spin()
            now = time.monotonic()
            runtime = state.snapshot()
            if planner.start_error:
                return "planner_start_failed:" + planner.start_error, last_frame_s, travelled_m
            if state.odom is not None:
                if last_pose is not None:
                    travelled_m += math.hypot(state.odom.x_m - last_pose.x_m, state.odom.y_m - last_pose.y_m)
                last_pose = state.odom
            frame, camera_error = camera.latest()
            if camera_error:
                return "camera_failed:" + camera_error, last_frame_s, travelled_m
            for result in drain_results(planner):
                event = {
                    "step": task_step,
                    "frame_to_result_ms": round((result.completed_s - result.job.frame.captured_monotonic_s) * 1000, 3),
                    "planner_worker_ms": round((result.completed_s - result.started_s) * 1000, 3),
                    "error": result.error,
                }
                planning_events.append(event)
                if result.error:
                    failure_count += 1
                    if failure_count >= loop_config.max_consecutive_plan_failures:
                        return "planner_failure:" + result.error, last_frame_s, travelled_m
                    continue
                assert result.planning is not None
                pixel_bbox = result.planning.get("pixel_bbox")
                if (
                    not tactical_bbox_annotated
                    and isinstance(pixel_bbox, list)
                    and len(pixel_bbox) == 4
                    and all(isinstance(value, (int, float)) for value in pixel_bbox)
                ):
                    tactical_bbox_annotated = True
                    annotated_path = annotate_tactical_view(
                        run_dir,
                        task_step=task_step,
                        pixel_bbox=pixel_bbox,
                    )
                    event["tactical_bbox"] = {
                        "pixel_bbox": [int(value) for value in pixel_bbox],
                        "image": str(annotated_path) if annotated_path is not None else None,
                    }
                if result.planning.get("final_stop_no_bbox"):
                    return "final_stop_no_bbox", last_frame_s, travelled_m
                if result.planning.get("va_requested_stop") and not result.job.allow_va_stop_for_final_approach:
                    va = result.planning.get("va")
                    bbox = va.get("bbox_2d") if isinstance(va, dict) else None
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        return "va_stop_no_bbox", last_frame_s, travelled_m
                    # Mirror the Baseline's final-approach branch without asking
                    # VA a second time: reuse the stop response and its bbox.
                    final_approach_requested = True
                    if not planner.submit(FrameJob(
                        result.job.frame,
                        result.job.origin,
                        artifact_prefix=result.job.artifact_prefix,
                        va_override=va,
                        allow_va_stop_for_final_approach=True,
                        final_stop_requested=terminate_after_approach,
                    )):
                        return "va_stop_final_approach_submit_failed", last_frame_s, travelled_m
                    continue
                event.update({
                    "timing_ms": result.planning.get("timing_ms"),
                    "goal_source": result.planning.get("goal_source"),
                    "infer_ms": result.planning.get("infer_ms"),
                })
                try:
                    if result.job.fixed_odom_goal is None:
                        reserved_path, fixed_odom_goal = reserve_baseline_execution_goal(
                            result.planning["trajectory"], origin=result.job.origin,
                            safe_distance_m=travel_policy.safe_distance_m,
                        )
                        visual_plan = admit_baseline_replan(
                            reserved_path, origin=result.job.origin,
                            captured_monotonic_s=result.job.frame.captured_monotonic_s,
                        )
                        event["fixed_execution_goal_odom"] = {
                            "x_m": fixed_odom_goal[0], "y_m": fixed_odom_goal[1],
                        }
                    else:
                        fixed_odom_goal = result.job.fixed_odom_goal
                        constrained_path, terminal_error_m = constrain_replan_to_fixed_goal(
                            result.planning["trajectory"], origin=result.job.origin,
                            goal_odom=fixed_odom_goal,
                            max_terminal_error_m=float(config["visual_closed_loop"]["max_replan_terminal_error_m"]),
                        )
                        visual_plan = admit_baseline_replan(
                            constrained_path, origin=result.job.origin,
                            captured_monotonic_s=result.job.frame.captured_monotonic_s,
                        )
                        event["replan_terminal_error_m"] = round(terminal_error_m, 4)
                        event["replan_endpoint_hard_constrained"] = True
                    if visual_plan.is_fresh(now, loop_config.max_plan_age_s):
                        follower = PathFollower(
                            visual_plan.odom_path,
                            PathFollowerConfig(
                                goal_tolerance_m=travel_policy.goal_tolerance_m,
                                max_duration_s=60.0,
                            ),
                        )
                        failure_count = 0
                    elif result.planning.get("goal_source") != "va_current_frame":
                        failure_count += 1
                except ValueError as exc:
                    if result.job.fixed_odom_goal is not None:
                        event["replan_rejected"] = str(exc)
                        failure_count += 1
                        if failure_count >= loop_config.max_consecutive_plan_failures:
                            return "planner_failure:" + str(exc), last_frame_s, travelled_m
                        continue
                    # G1 stops a local approach when its retained safety margin
                    # reaches the visual target. A pending LA/VA STOP converts
                    # that final local stop into whole-task termination.
                    prefix = (
                        "final_approach_completed:" if final_approach_requested
                        else "subgoal_stop_margin_reached:"
                    )
                    return prefix + str(exc), last_frame_s, travelled_m
            # Consume completed VA/plan responses before scheduling a new one.
            # In particular, this prevents a stale rolling replan from racing a
            # VA STOP response that needs its single final-approach plan.
            if (frame is not None and state.odom is not None
                    and frame.captured_monotonic_s > last_frame_s
                    and now - last_frame_s >= replan_interval_s):
                if planner.submit(FrameJob(
                    frame, state.odom,
                    artifact_prefix=f"step{task_step:03d}_replan",
                    final_stop_requested=terminate_after_approach,
                    fixed_odom_goal=fixed_odom_goal,
                )):
                    last_frame_s = frame.captured_monotonic_s
            if state.odom is not None and fixed_odom_goal is not None and baseline_goal_reached(
                state.odom, fixed_odom_goal, goal_tolerance_m=travel_policy.goal_tolerance_m,
            ):
                return (
                    "final_approach_completed:goal_tolerance_reached"
                    if final_approach_requested else "subgoal_stop_margin_reached:goal_tolerance_reached",
                    last_frame_s, travelled_m,
                )
            if follower is None or state.odom is None:
                desired_vx = desired_yaw = 0.0
                follower_data: dict[str, Any] = {"active": False, "stop_reason": "no_admitted_path"}
            else:
                output = follower.update(state.odom, now)
                desired_vx, desired_yaw = output.vx_mps, output.vyaw_radps
                follower_data = output.as_dict()
                if output.stop_reason:
                    if output.stop_reason == "goal_reached":
                        # A sampled NavDP path can end before the requested
                        # fixed goal. Keep G1's fixed-goal check authoritative.
                        follower = None
                    else:
                        return "follower_" + output.stop_reason, last_frame_s, travelled_m
            bridged = bridge.evaluate(
                desired_vx_mps=desired_vx, desired_vyaw_rps=desired_yaw, now_s=now,
                command_timestamp_s=now, runtime_state=runtime, motion_enabled=follower is not None,
            )
            sender.send(bridged.axes)
            control_audit.append({
                "step": task_step, "t_s": now, "travelled_m": travelled_m,
                "follower": follower_data, "bridge": asdict(bridged), "runtime_state": asdict(runtime),
            })
        return "mission_deadline_reached", last_frame_s, travelled_m
    finally:
        try:
            sender.send(Z1JoystickAxes.zero())
        finally:
            planner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/z1_planning.yaml"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--local-ip")
    parser.add_argument("--camera-source", choices=("sensor_manager", "direct"), default="sensor_manager")
    parser.add_argument(
        "--pose-source", choices=("chassis", "lidar_local"), default="chassis",
        help="yaw/pose source for turning and following; default: chassis (/odom_chassis)",
    )
    parser.add_argument("--max-duration-s", type=float, default=90.0)
    parser.add_argument(
        "--max-task-steps", type=int, default=12,
        help="maximum strategic TODO cycles in one mission (default: 12)",
    )
    parser.add_argument("--replan-interval-s", type=float, default=0.40)
    parser.add_argument("--horizon-m", type=float, default=0.30)
    parser.add_argument("--max-plan-age-s", type=float, default=1.0)
    parser.add_argument("--instruction", help="one-run task instruction; overrides config task.instruction")
    parser.add_argument("--target-description", help="one-run target description; overrides config task.target_description")
    parser.add_argument("--audit-full-chain", action="store_true",
                        help="in audit mode, call VA and NavDP after LA but never create an SDK sender")
    parser.add_argument("--artifact-dir", type=Path, default=Path("captures/missions/lavira"))
    args = parser.parse_args()
    enabled = args.execute and args.enable_motion
    if enabled and (args.confirm != CONFIRMATION or not args.local_ip):
        parser.error("execution requires --local-ip and exact --confirm " + CONFIRMATION)
    if args.max_duration_s <= 0.0 or args.replan_interval_s <= 0.0 or args.max_task_steps <= 0:
        parser.error("duration, replan interval, and max task steps must be positive")

    config = load_config(args.config)
    if args.instruction:
        config["task"] = dict(config["task"], instruction=args.instruction)
    if args.target_description:
        config["task"] = dict(config["task"], target_description=args.target_description)
    visual_config = config.get("visual_closed_loop", {})
    travel_policy = BaselineTravelPolicy(
        safe_distance_m=float(visual_config.get("baseline_safe_distance_m", 0.50)),
        goal_tolerance_m=float(visual_config.get("baseline_goal_tolerance_m", 1.00)),
    )
    loop_config = VisualLoopConfig(horizon_m=args.horizon_m, max_plan_age_s=args.max_plan_age_s)
    run_dir = args.artifact_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    # Keep every mission artifact in one self-contained run directory.
    config["artifacts"] = dict(config["artifacts"], directory=str(run_dir))

    state = Z1RosState(
        pose_source=args.pose_source,
        lidar_slam_config=LidarSlamHealthConfig(state_timeout_s=MISSION_LOCALIZATION_TIMEOUT_S),
    )
    bridge = Z1SafetyBridge(Z1SafetyBridgeConfig(
        odom_timeout_s=MISSION_ODOM_TIMEOUT_S,
        localization_timeout_s=MISSION_LOCALIZATION_TIMEOUT_S,
    ))
    video_recorder = MissionVideoRecorder(run_dir, float(config["camera"].get("fps", 15)))
    video_manifest: dict[str, Any] | None = None
    camera = CameraWorker(config["camera"], args.camera_source, video_recorder)
    sender: Z1SdkSender | None = None
    relay: RelayClient | None = None
    turn_audit: list[dict[str, Any]] = []
    control_audit: list[dict[str, Any]] = []
    planning_events: list[dict[str, Any]] = []
    task_steps: list[dict[str, Any]] = []
    audit_planning: dict[str, Any] | None = None
    panorama: dict[str, RGBDFrame] = {}
    la_panorama_bgr: dict[str, np.ndarray] = {}
    scan_frames: dict[str, RGBDFrame] = {}
    last_frame_s = float("-inf")
    panorama_origin_yaw_rad: float | None = None
    turn_config: HeadingTurnConfig | None = None
    stop_reason, la_payload, la_decision = None, None, None
    todo_list = "No TODO list yet. Please generate one."
    mission_deadline_s: float | None = None
    mission_start_base_pose: OdomPose | None = None
    mission_stop_base_pose: OdomPose | None = None
    try:
        camera.start()
        if enabled:
            sender = Z1SdkSender(args.local_ip)
            # SDK connect can take longer than the 0.20 s control watchdog.
            # Refresh odom/IMU/SLAM after it completes, immediately before any
            # panorama turn is permitted to send a non-zero joystick command.
            preflight(state)
        else:
            # Audit mode never turns.  It may still verify the live LA/VA API
            # using the forward view, and labels the reduced panorama honestly.
            deadline = time.monotonic() + 3.0
            while state.odom is None and time.monotonic() < deadline:
                state.spin()
                time.sleep(0.02)
        relay = RelayClient(config["relay"])
        relay.reset()
        state.spin()
        mission_start_base_pose = state.odom
        mission_deadline_s = time.monotonic() + args.max_duration_s
        for task_step in range(1, args.max_task_steps + 1):
            if time.monotonic() >= mission_deadline_s:
                stop_reason = "max_duration_reached"
                break

            # G1 captures a new panorama and asks LA again after each local
            # approach. Z1 recreates that panorama through a yaw scan.
            panorama, la_panorama_bgr, scan_frames = {}, {}, {}
            panorama["front"] = wait_for_frame(camera, last_frame_s)
            last_frame_s = panorama["front"].captured_monotonic_s
            save_frame(run_dir, f"step{task_step:03d}_panorama_front", panorama["front"])
            scan_frames["front_000deg"] = panorama["front"]
            la_panorama_bgr["front_000deg"] = labelled_observation(panorama["front"], "front_000deg")

            if enabled:
                assert sender is not None
                panorama_origin_yaw_rad = wait_for_settled_yaw(state)
                turn_config = HeadingTurnConfig(
                    yaw_tolerance_rad=TURN_CONTROL_TOLERANCE_RAD,
                    slowdown_error_rad=math.radians(12.0), timeout_s=12.0, max_yaw_rate_radps=0.20,
                )
                for name, direction in VIEW_TURNS:
                    outward, last_frame_s = scan_turn_steps(
                        state=state, bridge=bridge, sender=sender, camera=camera, direction=direction,
                        label=f"step{task_step:03d}_panorama_{name}", angles_deg=SCAN_STEP_DEGREES,
                        origin_yaw_rad=panorama_origin_yaw_rad, last_frame_s=last_frame_s,
                        output_dir=run_dir, audit=turn_audit, turn_config=turn_config,
                    )
                    scan_frames.update(outward)
                    panorama[name] = outward[f"step{task_step:03d}_panorama_{name}_090deg"]
                    for angle_deg in SCAN_STEP_DEGREES:
                        view_label = f"{name}_{angle_deg:03d}deg"
                        la_panorama_bgr[view_label] = labelled_observation(
                            outward[f"step{task_step:03d}_panorama_{name}_{angle_deg:03d}deg"], view_label,
                        )
                    drive_and_settle_to_yaw(
                        state, bridge, sender, target_yaw_rad=panorama_origin_yaw_rad,
                        phase=f"step{task_step:03d}_return_{name}_origin", audit=turn_audit, config=turn_config,
                    )

            offered_views = LA_SELECTABLE_VIEWS if enabled else ("front_000deg",)
            la_history = visual_history_info(task_steps)
            (run_dir / f"la_request_step{task_step:03d}.json").write_text(
                json.dumps({
                    "instruction": str(config["task"]["instruction"]),
                    "todo_list": todo_list,
                    "visual_history": la_history,
                    "step": task_step,
                    "is_initial": task_step == 1,
                    "selectable_views": list(offered_views),
                    "input_views": list(la_panorama_bgr),
                }, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            la_payload = relay.la_strategic(
                la_panorama_bgr,
                instruction=str(config["task"]["instruction"]), todo_list=todo_list,
                visual_history_info=la_history, step=task_step,
                is_initial=task_step == 1, selectable_views=offered_views,
            )
            la_decision = parse_strategic_decision(la_payload, selectable_views=set(offered_views))
            (run_dir / f"la_response_step{task_step:03d}.json").write_text(
                json.dumps(la_payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            if task_step == 1:
                (run_dir / "la_response.json").write_text(
                    json.dumps(la_payload, ensure_ascii=False, indent=2), encoding="utf-8",
                )
            if la_decision.todo_list:
                todo_list = la_decision.todo_list
            if la_decision.expected_landmark:
                config["task"] = dict(config["task"], target_description=la_decision.expected_landmark)
            if la_decision.progress_analysis:
                config["task"] = dict(config["task"], progress_analysis=la_decision.progress_analysis)
            step_audit: dict[str, Any] = {
                "step": task_step, "todo_list": todo_list,
                "selected_view": la_decision.selected_view,
                "expected_landmark": la_decision.expected_landmark,
                "progress_analysis": la_decision.progress_analysis,
                "la_action": la_decision.action,
                "scan_views": list(scan_frames),
            }
            task_steps.append(step_audit)

            if not enabled:
                if args.audit_full_chain:
                    pipeline = Z1PlanningPipeline(config, relay)
                    pipeline.adopt_session()
                    audit_planning = pipeline.run(panorama["front"], reset_session=False)
                    try:
                        reserved, fixed_goal = reserve_baseline_execution_goal(
                            audit_planning["trajectory"], origin=OdomPose(0.0, 0.0, 0.0),
                            safe_distance_m=travel_policy.safe_distance_m,
                        )
                        audit_planning["baseline_reserved_trajectory"] = reserved
                        audit_planning["baseline_fixed_execution_goal_odom"] = fixed_goal
                        audit_planning["baseline_control_admitted"] = True
                        step_audit["result"] = "audit_full_chain_completed_no_motion"
                    except ValueError as exc:
                        audit_planning["baseline_control_admitted"] = False
                        audit_planning["baseline_rejection"] = str(exc)
                        step_audit["result"] = "audit_full_chain_baseline_stop_margin"
                else:
                    step_audit["result"] = "audit_only_no_motion"
                stop_reason = step_audit["result"]
                break
            if la_decision.direction not in {"front", "left", "right", "behind"}:
                step_audit["result"] = "la_direction_not_supported_by_three_view_baseline"
                stop_reason = step_audit["result"]
                break

            assert sender is not None
            if la_decision.direction in {"left", "right", "behind"}:
                if panorama_origin_yaw_rad is None or turn_config is None:
                    raise RuntimeError("panorama_origin_yaw_missing")
                heading_direction = "right" if la_decision.direction == "behind" else la_decision.direction
                heading_degrees = 180 if la_decision.direction == "behind" else la_decision.turn_degrees
                heading_steps = tuple(range(30, heading_degrees + 1, 30))
                heading_frames, last_frame_s = scan_turn_steps(
                    state=state, bridge=bridge, sender=sender, camera=camera,
                    direction=heading_direction, label=f"step{task_step:03d}_la_heading_{la_decision.direction}",
                    angles_deg=heading_steps, origin_yaw_rad=panorama_origin_yaw_rad,
                    last_frame_s=last_frame_s, output_dir=run_dir, audit=turn_audit, turn_config=turn_config,
                )
                scan_frames.update(heading_frames)
            subgoal_result, last_frame_s, travelled_m = execute_subgoal(
                state=state, bridge=bridge, sender=sender, camera=camera, relay=relay, config=config,
                travel_policy=travel_policy, loop_config=loop_config, replan_interval_s=args.replan_interval_s,
                mission_deadline_s=mission_deadline_s, last_frame_s=last_frame_s, task_step=task_step,
                run_dir=run_dir,
                planning_events=planning_events, control_audit=control_audit,
                terminate_after_approach=la_decision.should_stop,
            )
            step_audit["result"] = subgoal_result
            step_audit["travelled_m"] = travelled_m
            if subgoal_result.startswith("subgoal_stop_margin_reached:"):
                continue
            if subgoal_result.startswith("final_approach_completed:"):
                stop_reason = "la_final_stop" if la_decision.should_stop else "va_final_stop"
                break
            if subgoal_result == "va_stop_no_bbox":
                stop_reason = "la_final_stop_no_bbox" if la_decision.should_stop else "va_final_stop_no_bbox"
                break
            if subgoal_result == "final_stop_no_bbox":
                stop_reason = "la_final_stop_no_bbox" if la_decision.should_stop else "va_final_stop_no_bbox"
                break
            stop_reason = subgoal_result
            break
        else:
            stop_reason = "max_task_steps_reached"

    except Exception as exc:
        stop_reason = "failed:" + str(exc)
    finally:
        mission_stop_base_pose = state.odom
        if sender is not None:
            try:
                sender.send(Z1JoystickAxes.zero())
                sender.close()
            except Exception as exc:
                stop_reason = stop_reason or "zero_command_failed:" + str(exc)
        camera.close()
        video_manifest = video_recorder.close()
        state.close()

    executed_odom_path_m = sum(float(step.get("travelled_m", 0.0)) for step in task_steps)
    base_stop_measurement: dict[str, Any] = {
        "pose_topic": state.pose_topic,
        "measurement": "odom_based_base_pose_not_external_ground_truth",
        "start_base_pose": asdict(mission_start_base_pose) if mission_start_base_pose else None,
        "stop_base_pose": asdict(mission_stop_base_pose) if mission_stop_base_pose else None,
        "subgoal_odom_path_length_m": executed_odom_path_m,
    }
    if mission_start_base_pose is not None and mission_stop_base_pose is not None:
        base_to_stop_displacement_m = math.hypot(
            mission_stop_base_pose.x_m - mission_start_base_pose.x_m,
            mission_stop_base_pose.y_m - mission_start_base_pose.y_m,
        )
        base_stop_measurement.update({
            "base_to_stop_straight_line_m": base_to_stop_displacement_m,
            "path_length_minus_straight_line_m": executed_odom_path_m - base_to_stop_displacement_m,
        })

    report = {
        "motion_commanded": bool(enabled), "uses_z1_sdk": bool(sender), "camera_source": args.camera_source,
        "pose_source": args.pose_source, "turn_yaw_source": state.pose_topic,
        "turn_control": {
            "panorama_origin_yaw_rad": panorama_origin_yaw_rad,
            "strategy": "absolute_origin_target_with_post_settle_residual_correction",
            "control_tolerance_degrees": math.degrees(TURN_CONTROL_TOLERANCE_RAD),
            "settle_duration_s": TURN_SETTLE_S,
            "settle_spread_degrees": math.degrees(TURN_SETTLE_SPREAD_RAD),
            "max_settle_corrections": TURN_MAX_SETTLE_CORRECTIONS,
        },
        "baseline_panorama_complete": set(panorama) == {"front", "right", "left"},
        "panorama_views": list(panorama), "scan_views": list(scan_frames),
        "la_input_views": list(la_panorama_bgr),
        "scan_step_degrees": 30, "baseline_travel_policy": asdict(travel_policy),
        "task": dict(config["task"]),
        "task_loop": {
            "max_task_steps": args.max_task_steps,
            "mission_deadline_s": mission_deadline_s,
            "final_todo_list": todo_list,
            "steps": task_steps,
        },
        "stop_reason": stop_reason, "la_decision": asdict(la_decision) if la_decision else None,
        "turn_audit": turn_audit, "planning_events": planning_events, "control_audit": control_audit,
        "audit_planning": audit_planning,
        "video": video_manifest,
        "base_stop_measurement": base_stop_measurement,
        "telemetry": {
            "lidar_local_tf": state.lidar_local_tf_samples,
            "lidar_slam_states": state.lidar_slam_state_samples,
        },
    }
    report_path = run_dir / "lavira_closed_loop_audit.json"
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"motion_commanded": enabled, "stop_reason": stop_reason,
                      "baseline_panorama_complete": report["baseline_panorama_complete"],
                      "audit_path": str(report_path)}, ensure_ascii=False, indent=2))
    return 0 if not str(stop_reason).startswith("failed:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
