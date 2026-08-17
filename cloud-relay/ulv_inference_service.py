"""
Thread-safe inference orchestration for the ULV relay server.

Owns DashScopeVisionClient (LA+VA) and IPlannerAdapter (trajectory).
Serializes reset/act operations per session with strict seq enforcement.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from dashscope_client import DashScopeVisionClient
from iplanner_adapter import IPlannerAdapter
from navdp_proxy import NavDPProxy

from prompts import (
    get_navigation_prompt_text,
    get_tactical_eyes_prompt,
    get_todo_generator_prompt,
)
from utils import safe_json_loads

PROTOCOL_VERSION = "magicbot.v2"
MAX_IMAGE_BYTES = 8 * 1024 * 1024


class SessionConflict(RuntimeError):
    pass


class SequenceConflict(RuntimeError):
    pass


def _image_bytes_to_base64(image_bytes: bytes, max_size: int = 512) -> str:
    """Convert JPEG bytes to base64 data URL, optionally resizing."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_image_content(base64_str: str) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
    }


class ULVInferenceService:
    """Stateful inference service: LA stragetic + VA tactical + iPlanner."""

    def __init__(
        self,
        dashscope_api_key: str,
        dashscope_model: str,
        iplanner_checkpoint: str,
        iplanner_config: str,
        iplanner_device: str = "cuda:0",
        navdp_url: str = "",
    ) -> None:
        self.dashscope = DashScopeVisionClient(
            api_key=dashscope_api_key,
            model=dashscope_model,
        )
        self.iplanner = None
        if iplanner_checkpoint and iplanner_config:
            self.iplanner = IPlannerAdapter(
                checkpoint_path=iplanner_checkpoint,
                config_path=iplanner_config,
                device=iplanner_device,
            )
        self.navdp = NavDPProxy(navdp_url) if navdp_url else None
        self._navdp_session: Optional[str] = None

        self.lock = threading.Lock()
        self.active_session: Optional[str] = None
        self.last_seq = -1
        self.total_steps = 0
        self.started_at = time.time()
        self.boot_id = uuid.uuid4().hex
        self._load_s = 0.0

    def health(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "boot_id": self.boot_id,
                "dashscope_model": self.dashscope.model_name,
                "iplanner_ready": self.iplanner is not None,
                "iplanner_device": self.iplanner.device if self.iplanner else "disabled",
                "navdp_ready": self.navdp.ready() if self.navdp else False,
                "uptime_s": round(time.time() - self.started_at, 1),
                "active_session": self.active_session,
                "last_seq": self.last_seq,
                "total_steps": self.total_steps,
                **self.dashscope.get_stats(),
            }

    def reset(self, session_id: str) -> Dict[str, Any]:
        if not session_id or len(session_id) > 128:
            raise ValueError("session_id must contain 1-128 characters")
        with self.lock:
            self.active_session = session_id
            self.last_seq = -1
            self._navdp_session = None
            return {
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "boot_id": self.boot_id,
                "session_id": session_id,
            }

    def la_strategic(
        self,
        session_id: str,
        seq: int,
        instruction: str,
        todo_list: str,
        visual_history_info: str,
        panorama_images: List[bytes],
        panorama_labels: List[str] | None = None,
        step: int = 1,
        is_initial: bool = False,
        selectable_views: List[str] | None = None,
    ) -> Dict[str, Any]:
        """LA strategic: 4-view panorama -> direction decision or STOP."""

        if not session_id:
            raise ValueError("session_id is required")
        if seq < 0:
            raise ValueError("seq must be non-negative")
        if len(panorama_images) < 1:
            raise ValueError("at least one panorama image required")
        if panorama_labels is not None and len(panorama_labels) != len(panorama_images):
            raise ValueError("panorama labels must match the image count")
        if selectable_views is not None:
            if not selectable_views or len(set(selectable_views)) != len(selectable_views):
                raise ValueError("selectable_views must contain distinct observation labels")
            if any(not view or len(view) > 64 for view in selectable_views):
                raise ValueError("selectable view label is invalid")

        with self.lock:
            if self.active_session != session_id:
                raise SessionConflict(
                    f"active session is {self.active_session!r}; call /v1/reset first"
                )
            if seq <= self.last_seq:
                raise SequenceConflict(
                    f"seq must increase: received {seq}, last accepted {self.last_seq}"
                )

            # Build LA prompt
            observation_labels = panorama_labels or [f"view_{i}" for i in range(len(panorama_images))]

            messages = [
                {
                    "role": "system",
                    "content": get_navigation_prompt_text(
                        instruction=instruction,
                        global_target="",
                        current_todo_list=todo_list,
                        history_info=visual_history_info,
                        current_step=step,
                        selectable_views=selectable_views,
                    ),
                }
            ]

            user_content: List[Dict[str, Any]] = []
            for i, img_bytes in enumerate(panorama_images):
                b64 = _image_bytes_to_base64(img_bytes)
                user_content.append({
                    "type": "text",
                    "text": f"\nObservation {observation_labels[i]}:",
                })
                user_content.append(_build_image_content(b64))

            messages.append({"role": "user", "content": user_content})

            t0 = time.perf_counter()
            content, info = self.dashscope.la_generate(
                messages, max_new_tokens=1024, temperature=0.0
            )
            api_latency_ms = (time.perf_counter() - t0) * 1000.0

            result = safe_json_loads(content)
            selected_view = ""
            if not bool(result.get("stop", False)) and selectable_views is not None:
                selected_view = str(result.get("selected_view", "")).strip().lower()
                if selected_view not in selectable_views:
                    raise ValueError("LA selected_view is not one of the offered observations")
            self.last_seq = seq
            self.total_steps += 1

        return {
            "ok": True,
            "protocol": PROTOCOL_VERSION,
            "boot_id": self.boot_id,
            "session_id": session_id,
            "seq": seq,
            "turn_direction": str(result.get("turn_direction", "front")),
            "selected_view": selected_view,
            "stop": bool(result.get("stop", False)),
            "updated_todo_list": str(result.get("updated_todo_list", todo_list)),
            "reasoning": str(result.get("reasoning", "")),
            "progress_analysis": str(result.get("progress_analysis", "")),
            "api_latency_ms": round(api_latency_ms, 1),
            "api_tokens": info.get("usage", {}),
        }

    def va_tactical(
        self,
        session_id: str,
        seq: int,
        instruction: str,
        target_description: str,
        progress_analysis: str,
        image_bytes: bytes,
    ) -> Dict[str, Any]:
        """VA tactical: single view -> bbox or STOP."""

        if not session_id:
            raise ValueError("session_id is required")
        if seq < 0:
            raise ValueError("seq must be non-negative")
        if not image_bytes:
            raise ValueError("image is empty")

        with self.lock:
            if self.active_session != session_id:
                raise SessionConflict(
                    f"active session is {self.active_session!r}; call /v1/reset first"
                )
            if seq <= self.last_seq:
                raise SequenceConflict(
                    f"seq must increase: received {seq}, last accepted {self.last_seq}"
                )

            b64 = _image_bytes_to_base64(image_bytes)
            messages = [
                {
                    "role": "system",
                    "content": get_tactical_eyes_prompt(
                        instruction=instruction,
                        global_target=target_description,
                        strategic_goal=target_description,
                        strategic_stop=False,
                        progress_analysis=progress_analysis,
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Current view:"},
                        _build_image_content(b64),
                    ],
                },
            ]

            t0 = time.perf_counter()
            content, info = self.dashscope.va_generate(
                messages, max_new_tokens=1024, temperature=0.0
            )
            api_latency_ms = (time.perf_counter() - t0) * 1000.0

            result = safe_json_loads(content)
            bbox = result.get("bbox_2d", [0, 0, 0, 0])
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox = [int(v) for v in bbox]
            else:
                bbox = [0, 0, 0, 0]

            self.last_seq = seq
            self.total_steps += 1

        return {
            "ok": True,
            "protocol": PROTOCOL_VERSION,
            "boot_id": self.boot_id,
            "session_id": session_id,
            "seq": seq,
            "action": str(result.get("action", "STOP")),
            "bbox_2d": bbox,
            "target": str(result.get("target", target_description)),
            "visual_check": str(result.get("visual_check", "")),
            "api_latency_ms": round(api_latency_ms, 1),
            "api_tokens": info.get("usage", {}),
        }

    def iplanner_plan(
        self,
        session_id: str,
        seq: int,
        goal_x: float,
        goal_y: float,
        depth_bytes: bytes,
    ) -> Dict[str, Any]:
        """iPlanner: RGB-depth + 2D goal -> trajectory."""

        if self.iplanner is None:
            return {
                "ok": False,
                "protocol": PROTOCOL_VERSION,
                "error": "iPlanner not available - checkpoint missing",
            }

        if not session_id:
            raise ValueError("session_id is required")
        if seq < 0:
            raise ValueError("seq must be non-negative")
        if not depth_bytes:
            raise ValueError("depth image is empty")

        with self.lock:
            if self.active_session != session_id:
                raise SessionConflict(
                    f"active session is {self.active_session!r}; call /v1/reset first"
                )

            # Decode depth from PNG (uint16 mm)
            with Image.open(io.BytesIO(depth_bytes)) as dimg:
                depth_mm = np.asarray(dimg, dtype=np.uint16)
            depth_m = depth_mm.astype(np.float32) / 1000.0

            t0 = time.perf_counter()
            trajectory = self.iplanner.plan(depth_m, (goal_x, goal_y))
            infer_ms = (time.perf_counter() - t0) * 1000.0

            self.last_seq = seq
            self.total_steps += 1

        if trajectory is None:
            trajectory = [[0.1, 0.0], [float(goal_x), float(goal_y)]]

        return {
            "ok": True,
            "protocol": PROTOCOL_VERSION,
            "boot_id": self.boot_id,
            "session_id": session_id,
            "seq": seq,
            "trajectory": trajectory,
            "infer_ms": round(infer_ms, 1),
        }

    def navdp_plan(
        self,
        session_id: str,
        seq: int,
        goal_x: float,
        goal_y: float,
        depth_scale_m: float,
        intrinsics: Dict[str, Any],
        color_bytes: bytes,
        depth_bytes: bytes,
    ) -> Dict[str, Any]:
        """Proxy RGB-D point-goal planning to the dedicated NavDP worker."""

        if self.navdp is None:
            return {"ok": False, "protocol": PROTOCOL_VERSION, "error": "NavDP worker is disabled"}
        if not session_id or seq < 0 or not color_bytes or not depth_bytes:
            raise ValueError("session_id, sequence, color image, and depth image are required")
        required_intrinsics = ("fx", "fy", "ppx", "ppy")
        if not all(name in intrinsics for name in required_intrinsics):
            raise ValueError("NavDP intrinsics are incomplete")

        with self.lock:
            if self.active_session != session_id:
                raise SessionConflict(
                    f"active session is {self.active_session!r}; call /v1/reset first"
                )
            if seq <= self.last_seq:
                raise SequenceConflict(
                    f"seq must increase: received {seq}, last accepted {self.last_seq}"
                )
            if self._navdp_session != session_id:
                self.navdp.reset(session_id)
                self._navdp_session = session_id
            result = self.navdp.plan(
                session_id=session_id,
                seq=seq,
                goal_x=goal_x,
                goal_y=goal_y,
                depth_scale_m=depth_scale_m,
                intrinsics=intrinsics,
                color_bytes=color_bytes,
                depth_bytes=depth_bytes,
            )
            self.last_seq = seq
            self.total_steps += 1

        return {
            "ok": True,
            "protocol": PROTOCOL_VERSION,
            "boot_id": self.boot_id,
            "session_id": session_id,
            "seq": seq,
            "trajectory": result["trajectory"],
            "infer_ms": result.get("infer_ms"),
            "memory_frames": result.get("memory_frames"),
            "candidate_count": result.get("candidate_count"),
            "selected_candidate_index": result.get("selected_candidate_index"),
            "selected_score": result.get("selected_score"),
        }
