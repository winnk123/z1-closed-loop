"""Strict HTTP client for the Uni-LaViRA Relay service over the SSH tunnel."""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any

import cv2
import numpy as np
import requests


class RelayError(RuntimeError):
    """A Relay response was unavailable, malformed, or not accepted."""


@dataclass(frozen=True)
class PlanResponse:
    trajectory: list[list[float]]
    infer_ms: float | None
    raw: dict[str, Any]
    planner: str = "iplanner"


class RelayClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = str(config["url"]).rstrip("/")
        self.timeout_s = float(config["timeout_s"])
        self.jpeg_quality = int(config["jpeg_quality"])
        self.navdp_upload_size = int(config.get("navdp_upload_size", 224))
        self.request_attempts = int(config.get("request_attempts", 3))
        self.retry_backoff_s = float(config.get("retry_backoff_s", 1.0))
        if self.navdp_upload_size <= 0:
            raise ValueError("navdp_upload_size must be positive")
        if self.request_attempts <= 0:
            raise ValueError("request_attempts must be positive")
        if self.retry_backoff_s < 0.0:
            raise ValueError("retry_backoff_s cannot be negative")
        self.session = requests.Session()
        self.session.trust_env = False
        self.session_id = f"z1-plan-{uuid.uuid4().hex[:16]}"
        self.boot_id = ""
        self.seq = 0

    def health(self) -> dict[str, Any]:
        return self._checked(self.session.get(f"{self.base_url}/health", timeout=self.timeout_s))

    def reset(self) -> dict[str, Any]:
        payload = self._checked(
            self.session.post(
                f"{self.base_url}/v1/reset",
                json={"session_id": self.session_id},
                timeout=self.timeout_s,
            )
        )
        if str(payload.get("session_id")) != self.session_id:
            raise RelayError("Relay reset returned a mismatched session_id")
        boot_id = str(payload.get("boot_id", ""))
        if not boot_id:
            raise RelayError("Relay reset response did not contain boot_id")
        self.boot_id = boot_id
        self.seq = 0
        return payload

    def va_tactical(
        self, color_bgr: np.ndarray, *, instruction: str, target_description: str, progress_analysis: str
    ) -> dict[str, Any]:
        started = perf_counter()
        image = self._jpeg(color_bgr)
        encoded = perf_counter()
        payload = self._post_multipart(
            "/v1/va_tactical",
            data={
                **self._request_identity(),
                "instruction": instruction,
                "target_description": target_description,
                "progress_analysis": progress_analysis,
            },
            files={"image": ("color.jpg", image, "image/jpeg")},
        )
        completed = perf_counter()
        payload["_client_timing_ms"] = {
            "image_encode": round((encoded - started) * 1000.0, 3),
            "http_round_trip": round((completed - encoded) * 1000.0, 3),
            "total": round((completed - started) * 1000.0, 3),
        }
        return payload

    def la_strategic(
        self,
        panorama_bgr: dict[str, np.ndarray],
        *,
        instruction: str,
        todo_list: str,
        visual_history_info: str,
        step: int,
        is_initial: bool,
        selectable_views: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Request one validated Baseline-style strategic LA decision.

        Every item is keyed by its offered observation label.  The label is
        rendered into its image by the caller and is also preserved in the
        multipart field name, so LA selects an observation rather than an
        abstract turn command.
        """
        if not panorama_bgr:
            raise ValueError("panorama must contain at least one labelled observation")
        if int(step) < 1:
            raise ValueError("LA step must be positive")
        if selectable_views is not None and not selectable_views:
            raise ValueError("selectable LA views cannot be empty")
        if selectable_views is None and not set(panorama_bgr).issubset({"front", "right", "left"}):
            raise ValueError("LA panorama must use named physical views")
        if selectable_views is not None and set(panorama_bgr) != set(selectable_views):
            raise ValueError("LA images must exactly match the offered observation labels")
        started = perf_counter()
        files: dict[str, tuple[str, bytes, str]] = {}
        for name, image in panorama_bgr.items():
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"LA {name} image must be an HxWx3 BGR image")
            files[f"image_{name}"] = (f"{name}.jpg", self._jpeg(image), "image/jpeg")
        encoded = perf_counter()
        payload = self._post_multipart(
            "/v1/la_strategic",
            data={
                **self._request_identity(),
                "instruction": instruction,
                "todo_list": todo_list,
                "visual_history_info": visual_history_info,
                "step": str(int(step)),
                "is_initial": "true" if is_initial else "false",
                "selectable_views": json.dumps(list(selectable_views or ())),
            },
            files=files,
        )
        # The deployed ULV relay encodes the Baseline action as ``stop: bool``
        # while the original client returns ``action: NAVIGATE|STOP``.  Accept
        # only that documented boolean fallback; every other form remains for
        # the strict strategic parser to reject before it can cause a turn.
        if "action" not in payload:
            stop = payload.get("stop")
            if not isinstance(stop, bool):
                raise RelayError("LA response must contain action or boolean stop")
            payload["action"] = "STOP" if stop else "NAVIGATE"
        completed = perf_counter()
        payload["_client_timing_ms"] = {
            "image_encode": round((encoded - started) * 1000.0, 3),
            "http_round_trip": round((completed - encoded) * 1000.0, 3),
            "total": round((completed - started) * 1000.0, 3),
        }
        return payload

    def iplanner_plan(self, depth_mm: np.ndarray, *, goal_x: float, goal_y: float) -> PlanResponse:
        if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            raise ValueError("aligned depth must be a uint16 HxW image in millimetres")
        started = perf_counter()
        ok, encoded = cv2.imencode(".png", depth_mm)
        if not ok:
            raise RelayError("failed to PNG-encode depth image")
        encoded_at = perf_counter()
        payload = self._post_multipart(
            "/v1/iplanner_plan",
            data={
                **self._request_identity(),
                "goal_x": f"{goal_x:.6f}",
                "goal_y": f"{goal_y:.6f}",
            },
            files={"depth_image": ("aligned_depth_mm.png", encoded.tobytes(), "image/png")},
        )
        completed = perf_counter()
        payload["_client_timing_ms"] = {
            "depth_encode": round((encoded_at - started) * 1000.0, 3),
            "http_round_trip": round((completed - encoded_at) * 1000.0, 3),
            "total": round((completed - started) * 1000.0, 3),
        }
        return self._plan_response(payload, planner="iplanner")

    def navdp_plan(self, color_bgr: np.ndarray, depth_mm: np.ndarray, *, goal_x: float, goal_y: float,
                   intrinsics: Any, depth_scale_m: float) -> PlanResponse:
        """Request a NavDP RGB-D local plan in the forward/left path convention."""
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("aligned color must be an HxWx3 BGR image")
        if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
            raise ValueError("aligned depth must be a uint16 HxW image in millimetres")
        if color_bgr.shape[:2] != depth_mm.shape:
            raise ValueError("aligned color and depth must have matching dimensions")
        started = perf_counter()
        color_input, depth_input, navdp_intrinsics = self._navdp_model_input(
            color_bgr, depth_mm, intrinsics,
        )
        prepared_at = perf_counter()
        color = self._jpeg(color_input)
        color_encoded_at = perf_counter()
        ok, encoded_depth = cv2.imencode(".png", depth_input)
        if not ok:
            raise RelayError("failed to PNG-encode depth image")
        depth_encoded_at = perf_counter()
        payload = self._post_multipart(
            "/v1/navdp_plan",
            data={
                **self._request_identity(),
                "goal_x": f"{goal_x:.6f}",
                "goal_y": f"{goal_y:.6f}",
                "depth_scale_m": f"{depth_scale_m:.9f}",
                "intrinsics": json.dumps(navdp_intrinsics),
            },
            files={
                "color_image": ("aligned_color.jpg", color, "image/jpeg"),
                "depth_image": ("aligned_depth_mm.png", encoded_depth.tobytes(), "image/png"),
            },
        )
        completed = perf_counter()
        payload["_client_timing_ms"] = {
            "model_input_prepare": round((prepared_at - started) * 1000.0, 3),
            "color_encode": round((color_encoded_at - prepared_at) * 1000.0, 3),
            "depth_encode": round((depth_encoded_at - color_encoded_at) * 1000.0, 3),
            "http_round_trip": round((completed - depth_encoded_at) * 1000.0, 3),
            "total": round((completed - started) * 1000.0, 3),
        }
        return self._plan_response(payload, planner="navdp")

    def _plan_response(self, payload: dict[str, Any], *, planner: str) -> PlanResponse:
        trajectory = payload.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise RelayError("Relay returned an invalid trajectory")
        normalized_trajectory: list[list[float]] = []
        for point in trajectory:
            if not isinstance(point, list) or len(point) < 2:
                raise RelayError("Relay returned an invalid trajectory point")
            try:
                normalized = [float(value) for value in point]
            except (TypeError, ValueError) as exc:
                raise RelayError("Relay trajectory contains a non-numeric point") from exc
            if not all(math.isfinite(value) for value in normalized):
                raise RelayError("Relay trajectory contains a non-finite point")
            normalized_trajectory.append(normalized)
        return PlanResponse(
            trajectory=normalized_trajectory,
            infer_ms=self._number_or_none(payload.get("infer_ms")),
            raw=payload,
            planner=planner,
        )

    def _request_identity(self) -> dict[str, str]:
        if not self.boot_id:
            raise RelayError("call reset() before requesting inference")
        current = self.seq
        self.seq += 1
        return {"session_id": self.session_id, "seq": str(current), "boot_id": self.boot_id}

    def _post_multipart(self, route: str, *, data: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> dict[str, Any]:
        last_error: RelayError | None = None
        for attempt in range(self.request_attempts):
            try:
                response = self.session.post(
                    f"{self.base_url}{route}", data=data, files=files, timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = RelayError(f"Relay request failed: {exc}")
            else:
                if response.status_code < 500:
                    payload = self._checked(response)
                    self._validate_identity(payload, expected_seq=str(data["seq"]))
                    return payload
                try:
                    self._checked(response)
                except RelayError as exc:
                    last_error = exc
                else:
                    last_error = RelayError(f"Relay HTTP {response.status_code}")
            if attempt + 1 < self.request_attempts:
                sleep(self.retry_backoff_s * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _checked(self, response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RelayError(f"Relay HTTP {response.status_code} returned non-JSON") from exc
        if not response.ok or not payload.get("ok", False):
            raise RelayError(f"Relay HTTP {response.status_code}: {payload.get('error', json.dumps(payload))}")
        return payload

    def _validate_identity(self, payload: dict[str, Any], *, expected_seq: str) -> None:
        if str(payload.get("session_id")) != self.session_id:
            raise RelayError("Relay session_id mismatch")
        if str(payload.get("boot_id")) != self.boot_id:
            raise RelayError("Relay boot_id changed; reset the session")
        if str(payload.get("seq")) != expected_seq:
            raise RelayError("Relay sequence mismatch")

    def _jpeg(self, image_bgr: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RelayError("failed to JPEG-encode color image")
        return encoded.tobytes()

    def _navdp_model_input(self, color_bgr: np.ndarray, depth_mm: np.ndarray, intrinsics: Any) -> tuple[
        np.ndarray, np.ndarray, dict[str, float | int]
    ]:
        """Apply NavDP's native 224-square letterbox before crossing the tunnel."""
        height, width = color_bgr.shape[:2]
        size = self.navdp_upload_size
        scale = size / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        color = cv2.resize(color_bgr, (resized_width, resized_height))
        depth = cv2.resize(depth_mm, (resized_width, resized_height))
        left = max((size - resized_width) // 2, 0)
        right = max(size - resized_width - left, 0)
        top = max((size - resized_height) // 2, 0)
        bottom = max(size - resized_height - top, 0)
        color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return color, depth, {
            "width": int(size), "height": int(size),
            "fx": float(intrinsics.fx) * scale, "fy": float(intrinsics.fy) * scale,
            "ppx": float(intrinsics.ppx) * scale + left,
            "ppy": float(intrinsics.ppy) * scale + top,
        }

    @staticmethod
    def _number_or_none(value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None
