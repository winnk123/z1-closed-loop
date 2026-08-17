#!/usr/bin/env python3
"""Persistent NavDP RGB-D worker for the Uni-LaViRA Relay host."""
from __future__ import annotations

import argparse
import cgi
import io
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

MAX_IMAGE_BYTES = 8 * 1024 * 1024


class NavDPWorker:
    def __init__(self, navdp_root: Path, checkpoint: Path, device: str) -> None:
        sys.path.insert(0, str(navdp_root))
        from policy_agent import NavDP_Agent

        self.agent = NavDP_Agent(np.eye(3, dtype=np.float32), navi_model=str(checkpoint), device=device)
        self.lock = threading.Lock()
        self.session_id: str | None = None
        self.last_seq = -1
        self.device = device

    def health(self) -> dict[str, Any]:
        return {"ok": True, "ready": True, "active_session": self.session_id, "last_seq": self.last_seq}

    def reset(self, session_id: str) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required")
        with self.lock:
            self.session_id = session_id
            self.last_seq = -1
            self.agent.reset(batch_size=1, threshold=-3.0)
        return {"ok": True, "session_id": session_id}

    def plan(self, session_id: str, seq: int, goal_x: float, goal_y: float, depth_scale_m: float,
             intrinsics: dict[str, Any], color_bytes: bytes, depth_bytes: bytes) -> dict[str, Any]:
        if not session_id or not color_bytes or not depth_bytes:
            raise ValueError("session_id, color_image, and depth_image are required")
        if not 0.00001 < depth_scale_m < 0.1:
            raise ValueError("invalid depth_scale_m")
        with self.lock:
            if session_id != self.session_id:
                raise ValueError("NavDP session is not reset")
            if seq <= self.last_seq:
                raise ValueError("NavDP sequence must increase")
            rgb = np.asarray(Image.open(io.BytesIO(color_bytes)).convert("RGB"))
            depth_mm = np.asarray(Image.open(io.BytesIO(depth_bytes)), dtype=np.uint16)
            if rgb.shape[:2] != depth_mm.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("color/depth alignment mismatch")
            self.agent.image_intrinsic = np.array([
                [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            goal = np.array([[goal_x, goal_y, 0.0]], dtype=np.float32)
            depth_m = depth_mm.astype(np.float32) * depth_scale_m
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            selected, all_trajectories, values, _ = self.agent.step_pointgoal(
                goal, rgb[None], depth_m[..., None][None]
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - started) * 1000.0
            self.last_seq = seq
            scores = values[0]
            selected_index = int(np.argmax(scores))
            trajectory = selected[0].tolist()
            # The controller path frame starts at the robot origin. NavDP's
            # first predicted point is already ahead of it.
            if not trajectory or np.linalg.norm(np.asarray(trajectory[0][:2], dtype=np.float32)) > 1e-6:
                trajectory.insert(0, [0.0, 0.0, 0.0])
            return {
                "ok": True,
                "session_id": session_id,
                "seq": seq,
                "trajectory": trajectory,
                "infer_ms": round(infer_ms, 1),
                "memory_frames": len(self.agent.memory_queue[0]),
                "candidate_count": int(all_trajectories.shape[1]),
                "selected_candidate_index": selected_index,
                "selected_score": float(scores[selected_index]),
            }


class Handler(BaseHTTPRequestHandler):
    server_version = "NavDPWorker/1.0"

    @property
    def worker(self) -> NavDPWorker:
        return self.server.worker  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, flush=True)

    def _respond(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(HTTPStatus.OK, self.worker.health())
        else:
            self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})

    def do_POST(self) -> None:
        try:
            if self.path == "/v1/reset":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._respond(HTTPStatus.OK, self.worker.reset(str(payload.get("session_id", ""))))
                return
            if self.path != "/v1/navdp_plan":
                self._respond(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})
                return
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                         "CONTENT_LENGTH": self.headers.get("Content-Length", "0")},
            )
            color = form["color_image"].file.read(MAX_IMAGE_BYTES + 1)
            depth = form["depth_image"].file.read(MAX_IMAGE_BYTES + 1)
            if len(color) > MAX_IMAGE_BYTES or len(depth) > MAX_IMAGE_BYTES:
                raise ValueError("input image exceeds size limit")
            result = self.worker.plan(
                str(form.getfirst("session_id", "")), int(form.getfirst("seq", "0")),
                float(form.getfirst("goal_x", "0")), float(form.getfirst("goal_y", "0")),
                float(form.getfirst("depth_scale_m", "0")), json.loads(form.getfirst("intrinsics", "{}")), color, depth,
            )
            self._respond(HTTPStatus.OK, result)
        except Exception as exc:
            self._respond(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navdp-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18889)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.worker = NavDPWorker(args.navdp_root, args.checkpoint, args.device)  # type: ignore[attr-defined]
    print(f"NavDP worker ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
