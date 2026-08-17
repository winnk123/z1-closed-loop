#!/usr/bin/env python3
"""ULV (Uni-LaViRA) remote inference relay server.

Bridges Go2 robot cameras to DashScope Qwen API (LA+VA) and local iPlanner (GPU).
Provides HTTP endpoints over SSH tunnel, following the minicpm magicbot protocol pattern.
"""
from __future__ import annotations

import argparse
import cgi
import json
import os
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from ulv_inference_service import (
    PROTOCOL_VERSION,
    MAX_IMAGE_BYTES,
    SessionConflict,
    SequenceConflict,
    ULVInferenceService,
)

MAX_JSON_BYTES = 64 * 1024


class ULVRequestHandler(BaseHTTPRequestHandler):
    server_version = "ULVInference/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(
            f"[http] {self.address_string()} [{self.log_date_time_string()}] "
            f"{fmt % args}\n"
        )
        sys.stdout.flush()

    @property
    def service(self) -> ULVInferenceService:
        return self.server.service  # type: ignore[attr-defined]

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json_response(status, {"ok": False, "protocol": PROTOCOL_VERSION, "error": message})

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > MAX_JSON_BYTES:
            raise ValueError(f"JSON body must contain 1-{MAX_JSON_BYTES} bytes")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _parse_multipart(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc

        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
            keep_blank_values=True,
        )

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(HTTPStatus.OK, self.service.health())
            return
        self._error(HTTPStatus.NOT_FOUND, "route not found")

    def do_POST(self) -> None:
        try:
            if self.path == "/v1/reset":
                payload = self._read_json()
                result = self.service.reset(str(payload.get("session_id", "")))
                self._json_response(HTTPStatus.OK, result)
                return
            if self.path == "/v1/la_strategic":
                self._handle_la_strategic()
                return
            if self.path == "/v1/va_tactical":
                self._handle_va_tactical()
                return
            if self.path == "/v1/iplanner_plan":
                self._handle_iplanner_plan()
                return
            if self.path == "/v1/navdp_plan":
                self._handle_navdp_plan()
                return
            self._error(HTTPStatus.NOT_FOUND, "route not found")
        except SessionConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except SequenceConflict as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def _handle_la_strategic(self) -> None:
        form = self._parse_multipart()
        session_id = str(form.getfirst("session_id", ""))
        seq = int(form.getfirst("seq", "0"))
        instruction = str(form.getfirst("instruction", "Follow the target person."))
        todo_list = str(form.getfirst("todo_list", "- [ ] Find target\n- [ ] Approach"))
        visual_history_info = str(form.getfirst("visual_history_info", ""))
        step = int(form.getfirst("step", "1"))
        is_initial = str(form.getfirst("is_initial", "false")).lower() == "true"
        raw_selectable_views = str(form.getfirst("selectable_views", "[]"))
        selectable_views = json.loads(raw_selectable_views)
        if not isinstance(selectable_views, list) or not all(isinstance(view, str) for view in selectable_views):
            raise ValueError("selectable_views must be a JSON string list")

        # The client sends one independently labelled image per candidate
        # observation.  Keep the offered order so the service can present the
        # matching label with every image.
        images = []
        image_keys = (
            [f"image_{view}" for view in selectable_views]
            if selectable_views else
            ["image_front", "image_right", "image_behind", "image_left"]
        )
        for key in image_keys:
            field = form[key] if key in form else None
            if field is not None and getattr(field, "file", None) is not None:
                data = field.file.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
                images.append(data)

        if not images:
            raise ValueError("at least one panorama image is required")

        result = self.service.la_strategic(
            session_id=session_id,
            seq=seq,
            instruction=instruction,
            todo_list=todo_list,
            visual_history_info=visual_history_info,
            panorama_images=images,
            panorama_labels=selectable_views or None,
            step=step,
            is_initial=is_initial,
            selectable_views=selectable_views,
        )
        self._json_response(HTTPStatus.OK, result)

    def _handle_va_tactical(self) -> None:
        form = self._parse_multipart()
        session_id = str(form.getfirst("session_id", ""))
        seq = int(form.getfirst("seq", "0"))
        instruction = str(form.getfirst("instruction", "Follow the target person."))
        target_description = str(form.getfirst("target_description", ""))
        progress_analysis = str(form.getfirst("progress_analysis", ""))

        image_field = form["image"] if "image" in form else None
        if image_field is None or getattr(image_field, "file", None) is None:
            raise ValueError("multipart field 'image' is required")
        image_bytes = image_field.file.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")

        result = self.service.va_tactical(
            session_id=session_id,
            seq=seq,
            instruction=instruction,
            target_description=target_description,
            progress_analysis=progress_analysis,
            image_bytes=image_bytes,
        )
        self._json_response(HTTPStatus.OK, result)

    def _handle_iplanner_plan(self) -> None:
        form = self._parse_multipart()
        session_id = str(form.getfirst("session_id", ""))
        seq = int(form.getfirst("seq", "0"))
        goal_x = float(form.getfirst("goal_x", "0.0"))
        goal_y = float(form.getfirst("goal_y", "0.0"))

        depth_field = form["depth_image"] if "depth_image" in form else None
        if depth_field is None or getattr(depth_field, "file", None) is None:
            raise ValueError("multipart field 'depth_image' is required")
        depth_bytes = depth_field.file.read(MAX_IMAGE_BYTES + 1)
        if len(depth_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(f"depth image exceeds {MAX_IMAGE_BYTES} bytes")

        result = self.service.iplanner_plan(
            session_id=session_id,
            seq=seq,
            goal_x=goal_x,
            goal_y=goal_y,
            depth_bytes=depth_bytes,
        )
        self._json_response(HTTPStatus.OK, result)

    def _handle_navdp_plan(self) -> None:
        form = self._parse_multipart()
        session_id = str(form.getfirst("session_id", ""))
        seq = int(form.getfirst("seq", "0"))
        goal_x = float(form.getfirst("goal_x", "0.0"))
        goal_y = float(form.getfirst("goal_y", "0.0"))
        depth_scale_m = float(form.getfirst("depth_scale_m", "0.0"))
        intrinsics = json.loads(str(form.getfirst("intrinsics", "{}")))
        if not isinstance(intrinsics, dict):
            raise ValueError("intrinsics must be a JSON object")
        color_field = form["color_image"] if "color_image" in form else None
        depth_field = form["depth_image"] if "depth_image" in form else None
        if color_field is None or depth_field is None:
            raise ValueError("color_image and depth_image are required")
        if getattr(color_field, "file", None) is None or getattr(depth_field, "file", None) is None:
            raise ValueError("NavDP image fields must be files")
        color_bytes = color_field.file.read(MAX_IMAGE_BYTES + 1)
        depth_bytes = depth_field.file.read(MAX_IMAGE_BYTES + 1)
        if len(color_bytes) > MAX_IMAGE_BYTES or len(depth_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("NavDP image exceeds size limit")
        result = self.service.navdp_plan(
            session_id=session_id,
            seq=seq,
            goal_x=goal_x,
            goal_y=goal_y,
            depth_scale_m=depth_scale_m,
            intrinsics=intrinsics,
            color_bytes=color_bytes,
            depth_bytes=depth_bytes,
        )
        self._json_response(HTTPStatus.OK, result)


class ULVHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple, service: ULVInferenceService) -> None:
        super().__init__(server_address, ULVRequestHandler)
        self.service = service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--dashscope-api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--dashscope-model", default="qwen3.5-27b")
    parser.add_argument("--iplanner-checkpoint", type=Path, default=None)
    parser.add_argument("--iplanner-config", type=Path, default=None)
    parser.add_argument("--iplanner-device", default="cuda:0")
    parser.add_argument("--navdp-url", default="http://127.0.0.1:18889")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.dashscope_api_key:
        print("[server] ERROR: DASHSCOPE_API_KEY not set", flush=True)
        sys.exit(1)

    iplanner_checkpoint = str(args.iplanner_checkpoint) if args.iplanner_checkpoint and args.iplanner_checkpoint.exists() else None

    iplanner_config = args.iplanner_config
    if iplanner_config is None and iplanner_checkpoint:
        iplanner_config = args.iplanner_checkpoint.parent.parent / "configs" / "iplanner.yaml"
    if iplanner_config and not iplanner_config.exists():
        iplanner_config = None

    if iplanner_checkpoint:
        print(
            f"[server] Starting DashScope model={args.dashscope_model} "
            f"iplanner checkpoint={iplanner_checkpoint} device={args.iplanner_device}",
            flush=True,
        )
    else:
        print(
            f"[server] Starting DashScope model={args.dashscope_model} "
            f"(iPlanner disabled - no checkpoint found)",
            flush=True,
        )

    service = ULVInferenceService(
        dashscope_api_key=args.dashscope_api_key,
        dashscope_model=args.dashscope_model,
        iplanner_checkpoint=iplanner_checkpoint,
        iplanner_config=str(iplanner_config) if iplanner_config else "",
        iplanner_device=args.iplanner_device,
        navdp_url=args.navdp_url,
    )

    server = ULVHTTPServer((args.host, args.port), service)
    print(
        f"[server] Ready http://{args.host}:{args.port} "
        f"boot_id={service.boot_id}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[server] Stopped", flush=True)


if __name__ == "__main__":
    main()
