"""HTTP proxy from the Relay process to the separately hosted NavDP worker."""
from __future__ import annotations

import json
from typing import Any

import requests


class NavDPProxy:
    def __init__(self, url: str, timeout_s: float = 10.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.trust_env = False

    def ready(self) -> bool:
        try:
            response = self.session.get(f"{self.url}/health", timeout=1.0)
            return response.ok and bool(response.json().get("ready"))
        except (requests.RequestException, ValueError):
            return False

    def reset(self, session_id: str) -> None:
        payload = self._request("/v1/reset", json={"session_id": session_id})
        if str(payload.get("session_id", "")) != session_id:
            raise RuntimeError("NavDP worker reset returned a mismatched session")

    def plan(self, *, session_id: str, seq: int, goal_x: float, goal_y: float,
             depth_scale_m: float, intrinsics: dict[str, Any], color_bytes: bytes, depth_bytes: bytes) -> dict[str, Any]:
        return self._request(
            "/v1/navdp_plan",
            data={
                "session_id": session_id, "seq": str(seq), "goal_x": f"{goal_x:.6f}",
                "goal_y": f"{goal_y:.6f}", "depth_scale_m": f"{depth_scale_m:.9f}",
                "intrinsics": json.dumps(intrinsics),
            },
            files={
                "color_image": ("color.jpg", color_bytes, "image/jpeg"),
                "depth_image": ("depth_mm.png", depth_bytes, "image/png"),
            },
        )

    def _request(self, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.post(f"{self.url}{path}", timeout=self.timeout_s, **kwargs)
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"NavDP worker unavailable: {exc}") from exc
        if not response.ok or not payload.get("ok", False):
            raise RuntimeError(f"NavDP worker rejected request: {payload.get('error', payload)}")
        return payload
