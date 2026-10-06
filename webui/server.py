#!/usr/bin/env python3
"""Standalone LF3R tactile WebUI server.

Serves the split tactile UI plus the fail-recovery manifest, synchronized tactile
series/sprites, and camera videos. No LF3R annotation/analysis services are
required.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from tactile_service import FailRecoveryTactileService


class TactileApplication:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.static_root = (Path(__file__).resolve().parent / "static").resolve()
        self.manifest_path = (
            self.root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "v1"
            / "failrecovery_manifest.jsonl"
        )
        self.tactile = FailRecoveryTactileService(self.root)
        self._rollouts: list[dict[str, Any]] | None = None
        self._rollout_map: dict[str, dict[str, Any]] | None = None

    def project_file(self, value: Any, suffix: str | None = None) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing project-relative path")
        path = (self.root / value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes repository root") from exc
        if suffix and path.suffix.lower() != suffix.lower():
            raise ValueError(f"expected {suffix} file")
        return path

    def load_rollouts(self) -> list[dict[str, Any]]:
        if self._rollouts is not None:
            return self._rollouts
        rows: list[dict[str, Any]] = []
        if self.manifest_path.is_file():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        continue
                    row = dict(value)
                    row.setdefault("manifest_source", str(self.manifest_path.relative_to(self.root)))
                    row.setdefault("manifest_label", "failrecovery")
                    rows.append(row)
        self._rollouts = rows
        self._rollout_map = {
            str(row.get("id")): row for row in rows if str(row.get("id") or "").strip()
        }
        return rows

    def rollout_map(self) -> dict[str, dict[str, Any]]:
        if self._rollout_map is None:
            self.load_rollouts()
        return self._rollout_map or {}


class TactileHandler(BaseHTTPRequestHandler):
    app: TactileApplication
    protocol_version = "HTTP/1.0"
    server_version = "LF3RTactile/1.0"

    CLIENT_DISCONNECT_ERRORS = (
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
    )

    def log_message(self, fmt: str, *args: Any) -> None:
        super().log_message(fmt, *args)

    def _safe_write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            return True
        except self.CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True
            return False

    def _finish_headers(self) -> bool:
        self.close_connection = True
        try:
            self.end_headers()
            return True
        except self.CLIENT_DISCONNECT_ERRORS:
            return False

    def json_response(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if self._finish_headers():
            self._safe_write(body)

    def json_error(self, status: int, message: str) -> None:
        self.json_response(status, {"error": message})

    def serve_static(self, path: Path) -> None:
        if not path.is_file():
            self.json_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if self._finish_headers():
            self._safe_write(body)

    def serve_png(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.send_header("Connection", "close")
        if self._finish_headers():
            self._safe_write(body)

    def serve_video(self, path: Path) -> None:
        if not path.is_file():
            self.json_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            left, right = match.groups()
            if left:
                start = int(left)
                end = int(right) if right else end
            elif right:
                count = int(right)
                start = max(size - count, 0)
            if start >= size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Connection", "close")
        if not self._finish_headers():
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk or not self._safe_write(chunk):
                    break
                remaining -= len(chunk)

    @staticmethod
    def _camera_for_rollout(rollout: dict[str, Any], query: dict[str, list[str]]) -> str:
        camera_paths = rollout.get("camera_video_paths")
        if not isinstance(camera_paths, dict) or not camera_paths:
            raise KeyError("Rollout has no camera videos")
        camera = str(query.get("camera", [""])[0] or "").strip()
        if camera and camera in camera_paths:
            return camera
        observation = rollout.get("observation_key")
        if isinstance(observation, str) and observation in camera_paths:
            return observation
        for preferred in ("cam_high", "cam_wrist", "cam_left_wrist", "cam_right_wrist"):
            if preferred in camera_paths:
                return preferred
        return str(next(iter(camera_paths)))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            if path == "/api/health":
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "manifest": str(self.app.manifest_path),
                        "rollouts": len(self.app.load_rollouts()),
                    },
                )
                return

            if path == "/api/rollouts":
                rows = self.app.load_rollouts()
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "rollouts": rows,
                        "manifests": [
                            {
                                "path": str(self.app.manifest_path.relative_to(self.app.root)),
                                "label": "failrecovery",
                                "rollouts": len(rows),
                                "valid": self.app.manifest_path.is_file(),
                            }
                        ],
                    },
                )
                return

            if path.startswith("/api/videos/"):
                rollout_id = path[len("/api/videos/"):].strip("/")
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                camera = self._camera_for_rollout(rollout, query)
                camera_paths = rollout.get("camera_video_paths") or {}
                value = camera_paths.get(camera)
                video = self.app.project_file(value, ".mp4")
                self.serve_video(video)
                return

            if path.startswith("/api/tactile/"):
                parts = path.strip("/").split("/")
                if len(parts) != 4 or parts[:2] != ["api", "tactile"]:
                    self.json_error(HTTPStatus.NOT_FOUND, "Tactile resource not found")
                    return
                rollout_id, resource = parts[2], parts[3]
                camera = str(query.get("camera", ["cam_high"])[0] or "cam_high")
                if resource == "series":
                    self.json_response(
                        HTTPStatus.OK,
                        {"tactile": self.app.tactile.series(rollout_id, camera)},
                    )
                    return
                if resource == "frame":
                    frame = int(query.get("frame", ["0"])[0])
                    self.json_response(
                        HTTPStatus.OK,
                        {"tactile": self.app.tactile.frame(rollout_id, camera, frame)},
                    )
                    return
                if resource == "sprite":
                    frame = int(query.get("frame", ["0"])[0])
                    kind = str(query.get("kind", ["deform"])[0] or "deform")
                    self.serve_png(self.app.tactile.sprite(rollout_id, camera, frame, kind))
                    return
                if resource == "image":
                    finger = str(query.get("finger", [""])[0] or "")
                    event_id = str(query.get("event_id", [""])[0] or "")
                    kind = str(query.get("kind", ["deform"])[0] or "deform")
                    self.serve_png(self.app.tactile.image(rollout_id, finger, event_id, kind))
                    return
                self.json_error(HTTPStatus.NOT_FOUND, "Tactile resource not found")
                return

            if path in {"", "/"}:
                self.serve_static(self.app.static_root / "tactile" / "index.html")
                return

            if path.startswith("/static/"):
                relative = path[len("/static/"):]
                target = (self.app.static_root / relative).resolve()
                try:
                    target.relative_to(self.app.static_root)
                except ValueError:
                    self.json_error(HTTPStatus.FORBIDDEN, "Invalid static path")
                    return
                self.serve_static(target)
                return

            self.json_error(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self.json_error(HTTPStatus.NOT_FOUND, str(exc.args[0]))
        except (ValueError, json.JSONDecodeError) as exc:
            self.json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except OSError as exc:
            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    app = TactileApplication(args.root)
    handler = type("BoundTactileHandler", (TactileHandler,), {"app": app})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"LF3R tactile WebUI: http://{args.host}:{args.port}/", flush=True)
    print(f"Dataset manifest: {app.manifest_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
