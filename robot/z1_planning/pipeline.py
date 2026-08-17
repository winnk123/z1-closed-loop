"""Single-frame, zero-motion image-to-waypoint pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .geometry import LocalGoal, bbox_to_local_goal
from .realsense_camera import RGBDFrame
from .relay_client import PlanResponse, RelayClient


class Z1PlanningPipeline:
    def __init__(self, config: dict[str, Any], relay: RelayClient) -> None:
        self.config = config
        self.relay = relay
        self.artifacts_dir = Path(str(config["artifacts"]["directory"]))
        self._session_ready = False
        self._session_health: dict[str, Any] | None = None

    def start_session(self) -> dict[str, Any]:
        """Check planner availability and initialize one persistent Relay session."""
        health = self.relay.health()
        planner_name = str(self.config["planning"].get("planner", "iplanner")).lower()
        ready_key = {"iplanner": "iplanner_ready", "navdp": "navdp_ready"}.get(planner_name)
        if ready_key is None:
            raise ValueError(f"unsupported planner: {planner_name}")
        if not health.get(ready_key, False):
            raise RuntimeError(f"Relay {planner_name} is not ready; refusing to fabricate a path")
        self.relay.reset()
        self._session_ready = True
        self._session_health = health
        return health

    def adopt_session(self) -> dict[str, Any]:
        """Adopt an already-reset relay session after LA has consumed seq 0."""
        if not self.relay.boot_id:
            raise RuntimeError("relay session must be reset before adoption")
        health = self.relay.health()
        planner_name = str(self.config["planning"].get("planner", "iplanner")).lower()
        ready_key = {"iplanner": "iplanner_ready", "navdp": "navdp_ready"}.get(planner_name)
        if ready_key is None or not health.get(ready_key, False):
            raise RuntimeError(f"Relay {planner_name} is not ready")
        self._session_ready = True
        self._session_health = health
        return health

    def run(
        self, frame: RGBDFrame, *, reset_session: bool = True,
        local_goal_override: LocalGoal | None = None,
        va_override: dict[str, Any] | None = None,
        allow_va_stop_for_final_approach: bool = False,
        final_stop_requested: bool = False,
        write_artifacts: bool = True,
        artifact_prefix: str = "planning",
    ) -> dict[str, Any]:
        """Request VA or reuse a fixed local target, then run the local planner.

        A normal VA ``STOP`` is returned to the task loop as a semantic stop
        signal, rather than being misclassified as a planner failure.  The
        task loop may then reuse that exact VA response for its one final
        approach by setting ``allow_va_stop_for_final_approach``.
        """
        started = perf_counter()
        timings_ms: dict[str, float | dict[str, float]] = {}
        if reset_session:
            health = self.start_session()
        elif not self._session_ready:
            raise RuntimeError("call start_session() before rolling visual planning")
        else:
            assert self._session_health is not None
            health = self._session_health
        planner_name = str(self.config["planning"].get("planner", "iplanner")).lower()
        task = self.config["task"]
        va: dict[str, Any] | None = None
        pixel_box: tuple[int, int, int, int] | None = None
        if local_goal_override is None:
            if va_override is None:
                va_started = perf_counter()
                va = self.relay.va_tactical(
                    frame.color_bgr,
                    instruction=str(task["instruction"]),
                    target_description=str(task["target_description"]),
                    progress_analysis=str(task["progress_analysis"]),
                )
                timings_ms["va_total"] = round((perf_counter() - va_started) * 1000.0, 3)
            else:
                va = va_override
            va_artifacts = (
                self._write_va_artifacts(frame, va, artifact_prefix)
                if write_artifacts else None
            )
            va_requested_stop = str(va.get("action", "STOP")).upper() != "NAVIGATE"
            if va_requested_stop and not allow_va_stop_for_final_approach:
                return {
                    "health": health,
                    "va": va,
                    "va_requested_stop": True,
                    "goal_source": "va_current_frame",
                    "timing_ms": timings_ms,
                    "artifact_dir": str(self.artifacts_dir) if va_artifacts is not None else None,
                    "artifact_files": va_artifacts,
                }
            bbox = va.get("bbox_2d")
            if not isinstance(bbox, list):
                if final_stop_requested or allow_va_stop_for_final_approach:
                    return {
                        "health": health,
                        "va": va,
                        "final_stop_no_bbox": True,
                        "goal_source": "va_current_frame",
                        "timing_ms": timings_ms,
                        "artifact_dir": str(self.artifacts_dir) if va_artifacts is not None else None,
                        "artifact_files": va_artifacts,
                    }
                raise RuntimeError("VA response did not contain bbox_2d")
            geometry_started = perf_counter()
            try:
                goal, pixel_box = bbox_to_local_goal(frame, bbox, self.config["planning"])
            except ValueError as exc:
                if final_stop_requested or allow_va_stop_for_final_approach:
                    return {
                        "health": health,
                        "va": va,
                        "final_stop_no_bbox": True,
                        "bbox_error": str(exc),
                        "goal_source": "va_current_frame",
                        "timing_ms": timings_ms,
                        "artifact_dir": str(self.artifacts_dir) if va_artifacts is not None else None,
                        "artifact_files": va_artifacts,
                    }
                raise
            timings_ms["goal_geometry"] = round((perf_counter() - geometry_started) * 1000.0, 3)
        else:
            goal = local_goal_override
        planner_started = perf_counter()
        if planner_name == "navdp":
            plan = self.relay.navdp_plan(
                frame.color_bgr,
                frame.depth_mm,
                goal_x=goal.x_forward_m,
                goal_y=goal.y_left_m,
                intrinsics=frame.intrinsics,
                depth_scale_m=frame.depth_scale_m,
            )
        else:
            plan = self.relay.iplanner_plan(
                frame.depth_mm, goal_x=goal.x_forward_m, goal_y=goal.y_left_m
            )
        timings_ms["planner_client_total"] = round((perf_counter() - planner_started) * 1000.0, 3)
        client_timing = plan.raw.get("_client_timing_ms")
        if isinstance(client_timing, dict):
            timings_ms["planner_client_breakdown"] = client_timing
        artifact_started = perf_counter()
        artifact_files = (
            self._write_plan_artifacts(frame, va, goal, pixel_box, plan, artifact_prefix)
            if write_artifacts else None
        )
        if write_artifacts:
            timings_ms["artifact_write"] = round((perf_counter() - artifact_started) * 1000.0, 3)
        timings_ms["pipeline_total"] = round((perf_counter() - started) * 1000.0, 3)
        return {
            "artifact_dir": str(self.artifacts_dir) if artifact_files is not None else None,
            "artifact_files": artifact_files,
            "health": health,
            "va": va,
            "va_requested_stop": bool(
                va is not None and str(va.get("action", "STOP")).upper() != "NAVIGATE"
            ),
            "goal": asdict(goal),
            "goal_source": "va_current_frame" if va is not None else "fixed_odom_target",
            "planner": plan.planner,
            "trajectory": plan.trajectory,
            "infer_ms": plan.infer_ms,
            "timing_ms": timings_ms,
        }

    def _write_va_artifacts(
        self,
        frame: RGBDFrame,
        va: dict[str, Any],
        artifact_prefix: str,
    ) -> dict[str, str]:
        """Persist every raw VA result before semantic-stop handling.

        A VA ``STOP`` or missing bbox is evidence, not an error path that may
        discard its response.  All files stay in the mission directory.
        """
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        color_path = self.artifacts_dir / f"{artifact_prefix}_input_color.jpg"
        depth_path = self.artifacts_dir / f"{artifact_prefix}_input_depth_mm.png"
        response_path = self.artifacts_dir / f"{artifact_prefix}_response.json"
        if not cv2.imwrite(str(color_path), frame.color_bgr):
            raise RuntimeError("failed to write VA input color artifact")
        if not cv2.imwrite(str(depth_path), frame.depth_mm):
            raise RuntimeError("failed to write VA input depth artifact")
        response_path.write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
            "camera_intrinsics": asdict(frame.intrinsics),
            "depth_scale_m": frame.depth_scale_m,
            "va": va,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "va_input_color": str(color_path),
            "va_input_depth": str(depth_path),
            "va_response": str(response_path),
        }

    def _write_plan_artifacts(
        self,
        frame: RGBDFrame,
        va: dict[str, Any] | None,
        goal: LocalGoal,
        pixel_box: tuple[int, int, int, int] | None,
        plan: PlanResponse,
        artifact_prefix: str,
    ) -> dict[str, str]:
        """Write planner output beside the VA evidence, never in a side tree."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        debug = frame.color_bgr.copy()
        if pixel_box is not None:
            left, top, right, bottom = pixel_box
            cv2.rectangle(debug, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.circle(debug, (goal.pixel_u, goal.pixel_v), 4, (0, 0, 255), -1)
        self._draw_path(debug, plan.trajectory, frame, goal)
        overlay_path = self.artifacts_dir / f"{artifact_prefix}_plan_overlay.jpg"
        if not cv2.imwrite(str(overlay_path), debug):
            raise RuntimeError("failed to write plan overlay artifact")
        report = {
            "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
            "motion_commanded": False,
            "camera_intrinsics": asdict(frame.intrinsics),
            "depth_scale_m": frame.depth_scale_m,
            "va": va,
            "pixel_bbox": list(pixel_box) if pixel_box is not None else None,
            "local_goal": asdict(goal),
            "planner": plan.planner,
            "trajectory": plan.trajectory,
            "infer_ms": plan.infer_ms,
            "timing_ms": plan.raw.get("_client_timing_ms"),
        }
        report_path = self.artifacts_dir / f"{artifact_prefix}_plan.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "plan_overlay": str(overlay_path),
            "plan_result": str(report_path),
        }

    def _draw_path(self, image: np.ndarray, trajectory: list[list[float]], frame: RGBDFrame, goal: LocalGoal) -> None:
        """Project ground-plane trajectory points with the G1 baseline convention."""
        intr = frame.intrinsics
        camera = self.config.get("camera", {})
        camera_height_m = float(camera.get("camera_height_m", 1.0))
        roll_correction = float(camera.get("camera_roll_correction", 0.0))
        points: list[tuple[int, int]] = []
        for waypoint in trajectory:
            if len(waypoint) < 2:
                continue
            forward, left = float(waypoint[0]), float(waypoint[1])
            # Robot X/Y maps to camera Z/-X. Ground is camera Y=height.
            if forward <= 0.01:
                continue
            camera_z = forward
            camera_x = -left
            camera_y = camera_height_m + camera_x * roll_correction
            u = int(round(camera_x * intr.fx / camera_z + intr.ppx))
            v = int(round(camera_y * intr.fy / camera_z + intr.ppy))
            if 0 <= u < intr.width and 0 <= v < intr.height:
                points.append((u, v))
        for first, second in zip(points, points[1:]):
            cv2.line(image, first, second, (255, 0, 0), 2)
