#!/usr/bin/env python3
"""Standalone tactile WebUI server.

The server intentionally keeps only the fail-recovery tactile dataset and the
SHARPA experiment surface.  It provides the same high-level workspace shape as
the dissertation WebUI (Annotate / Results / Runs / Analysis / Settings)
without importing the LF3R repair, LIBERO, baseline, or world-model backend.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

try:  # package import for tests
    from .tactile_service import FailRecoveryTactileService
except ImportError:  # direct ``python webui/server.py`` execution
    from tactile_service import FailRecoveryTactileService


EVENT_TYPES: dict[int, dict[str, str]] = {
    6: {"name": "align_failure", "label": "Align failure", "outcome": "failure", "phase": "align"},
    7: {"name": "insert_failure", "label": "Insert failure", "outcome": "failure", "phase": "insert"},
    8: {"name": "align_success", "label": "Align success", "outcome": "success", "phase": "align"},
    9: {"name": "insert_success", "label": "Insert success", "outcome": "success", "phase": "insert"},
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "source_project_root": ".",
    "annotations_path": "annotations/failrecovery/records",
    "annotation_seed_glob": "outputs/usb_event_intervals/*/intervals.jsonl",
    "runs_root": "outputs",
    "default_camera": "cam_high",
    "default_tactile_kind": "deform",
    "theme": "light",
}


class TactileApplication:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.static_root = (Path(__file__).resolve().parent / "static").resolve()
        self.settings_path = self.root / ".tactile_webui" / "settings.json"
        self._settings = self._load_settings()
        self._rollouts: list[dict[str, Any]] | None = None
        self._rollout_map: dict[str, dict[str, Any]] | None = None
        self._annotation_lock = threading.RLock()
        self._configure_data_source()

    # ---------- generic filesystem helpers ----------

    def _configure_data_source(self) -> None:
        raw = str(self._settings.get("source_project_root") or ".").strip()
        source = Path(raw).expanduser()
        if not source.is_absolute():
            source = self.root / source
        self.source_root = source.resolve()
        dataset_root = self.source_root / "datasets" / "failrecovery"
        manifest = dataset_root / "manifest.jsonl"
        transitional_manifest = (
            self.source_root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "failrecovery_manifest.jsonl"
        )
        legacy_manifest = (
            self.source_root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "v1"
            / "failrecovery_manifest.jsonl"
        )
        # New standalone layout is dataset-centric. The two old paths are
        # read-only compatibility inputs for repositories not migrated yet.
        if manifest.is_file():
            self.manifest_path = manifest
        elif transitional_manifest.is_file():
            self.manifest_path = transitional_manifest
        else:
            self.manifest_path = legacy_manifest
        self.tactile = FailRecoveryTactileService(self.source_root, self.manifest_path)
        self._rollouts = None
        self._rollout_map = None

    @staticmethod
    def _display_path(path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

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

    def source_file(self, value: Any, suffix: str | None = None) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing source-project-relative path")
        raw = Path(value).expanduser()
        path = raw.resolve() if raw.is_absolute() else (self.source_root / raw).resolve()
        try:
            path.relative_to(self.source_root)
        except ValueError as exc:
            raise ValueError("source data path escapes source project root") from exc
        if suffix and path.suffix.lower() != suffix.lower():
            raise ValueError(f"expected {suffix} file")
        return path

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object in {path}:{line_number}")
                rows.append(value)
        return rows

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    # ---------- settings ----------

    def _load_settings(self) -> dict[str, Any]:
        settings = dict(DEFAULT_SETTINGS)
        if self.settings_path.is_file():
            loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                settings.update({key: loaded[key] for key in DEFAULT_SETTINGS if key in loaded})
        return settings

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = dict(self._settings)
        for key in DEFAULT_SETTINGS:
            if key not in payload:
                continue
            value = payload[key]
            if key == "source_project_root":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("source_project_root must be a non-empty path")
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = self.root / candidate
                if not candidate.resolve().is_dir():
                    raise ValueError("source_project_root does not exist or is not a directory")
                settings[key] = value.strip()
            elif key in {"annotations_path", "annotation_seed_glob", "runs_root"}:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be a non-empty relative path")
                if Path(value).is_absolute() or ".." in Path(value).parts:
                    raise ValueError(f"{key} must remain inside the repository")
                settings[key] = value.strip()
            elif key == "default_camera":
                settings[key] = str(value or "cam_high").strip()
            elif key == "default_tactile_kind":
                if value not in {"raw", "deform"}:
                    raise ValueError("default_tactile_kind must be raw or deform")
                settings[key] = value
            elif key == "theme":
                if value not in {"light", "dark"}:
                    raise ValueError("theme must be light or dark")
                settings[key] = value
        source_changed = settings.get("source_project_root") != self._settings.get("source_project_root")
        self._settings = settings
        self._atomic_text(self.settings_path, json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
        if source_changed:
            self._configure_data_source()
        return self.settings

    # ---------- rollout manifest ----------

    def load_rollouts(self, refresh: bool = False) -> list[dict[str, Any]]:
        if self._rollouts is not None and not refresh:
            return self._rollouts
        rows: list[dict[str, Any]] = []
        if self.manifest_path.is_file():
            for value in self._read_jsonl(self.manifest_path):
                row = dict(value)
                row.setdefault("manifest_source", self._display_path(self.manifest_path, self.source_root))
                row.setdefault("manifest_label", "failrecovery")
                row.setdefault("source_kind", "real_robot")
                row.setdefault("task_description", row.get("instruction") or row.get("task_key") or "Tactile task")
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

    # ---------- tactile interval annotations ----------

    def annotation_target_path(self) -> Path:
        """Directory matching the original LF3R annotation record layout."""
        return self.project_file(self._settings["annotations_path"])

    def annotation_record_path(self, rollout_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", rollout_id):
            raise ValueError("invalid rollout id")
        return self.annotation_target_path() / f"{rollout_id}.tactile.json"

    def annotation_seed_path(self) -> Path | None:
        pattern = str(self._settings["annotation_seed_glob"])
        candidates = sorted(
            (path for path in self.source_root.glob(pattern) if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
        return candidates[-1] if candidates else None

    def _record_annotation_rows(self) -> list[dict[str, Any]]:
        records_dir = self.annotation_target_path()
        if not records_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(records_dir.glob("*.tactile.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            rollout_id = str(record.get("rollout_id") or path.name.removesuffix(".tactile.json"))
            intervals = record.get("tactile_intervals")
            if not isinstance(intervals, list):
                continue
            for raw in intervals:
                if isinstance(raw, dict):
                    item = dict(raw)
                    item["rollout_id"] = rollout_id
                    rows.append(item)
        return rows

    def load_annotation_rows(self) -> tuple[list[dict[str, Any]], Path | None]:
        with self._annotation_lock:
            seed = self.annotation_seed_path()
            rows = self._read_jsonl(seed) if seed is not None else []
            record_rows = self._record_annotation_rows()
            edited_rollouts = {str(row.get("rollout_id") or "") for row in record_rows}
            if edited_rollouts:
                rows = [row for row in rows if str(row.get("rollout_id") or "") not in edited_rollouts]
                rows.extend(record_rows)
                return rows, self.annotation_target_path()
            return rows, seed

    def annotations_by_rollout(self) -> dict[str, list[dict[str, Any]]]:
        rows, _ = self.load_annotation_rows()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            try:
                key = int(row.get("event_key"))
            except (TypeError, ValueError):
                continue
            rollout_id = str(row.get("rollout_id") or "")
            if rollout_id and key in EVENT_TYPES:
                item = dict(row)
                item["event_key"] = key
                item.setdefault("event_name", EVENT_TYPES[key]["name"])
                grouped[rollout_id].append(item)
        for items in grouped.values():
            items.sort(
                key=lambda event: (
                    int(event.get("start_frame", 0)),
                    int(event.get("end_frame", 0)),
                    int(event.get("event_index", 0)),
                )
            )
        return grouped

    def _validate_events(self, rollout_id: str, events: Any) -> list[dict[str, Any]]:
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        record = self.rollout_map().get(rollout_id)
        if not record:
            raise KeyError("Unknown rollout")
        total_frames = int(record.get("total_frames") or 0)
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(events):
            if not isinstance(raw, dict):
                raise ValueError(f"event {index + 1} must be an object")
            try:
                key = int(raw.get("event_key"))
                start = int(raw.get("start_frame"))
                end = int(raw.get("end_frame"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"event {index + 1} has invalid key/start/end") from exc
            if key not in EVENT_TYPES:
                raise ValueError(f"event {index + 1} has unsupported event_key {key}")
            if start < 0 or end < start:
                raise ValueError(f"event {index + 1} requires 0 <= start <= end")
            if total_frames and end >= total_frames:
                raise ValueError(f"event {index + 1} end {end} exceeds rollout length {total_frames}")
            event_index = index
            normalized.append(
                {
                    "event_id": f"{rollout_id}:event:{event_index}",
                    "rollout_id": rollout_id,
                    "event_index": event_index,
                    "event_key": key,
                    "event_name": EVENT_TYPES[key]["name"],
                    "start_frame": start,
                    "end_frame": end,
                    "notes": str(raw.get("notes") or ""),
                }
            )
        normalized.sort(key=lambda event: (event["start_frame"], event["end_frame"], event["event_key"]))
        for index, event in enumerate(normalized):
            event["event_index"] = index
            event["event_id"] = f"{rollout_id}:event:{index}"
        return normalized

    def save_rollout_annotations(self, rollout_id: str, events: Any) -> list[dict[str, Any]]:
        normalized = self._validate_events(rollout_id, events)
        target = self.annotation_record_path(rollout_id)
        with self._annotation_lock:
            previous: dict[str, Any] = {}
            if target.is_file():
                try:
                    loaded = json.loads(target.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        previous = loaded
                except (OSError, json.JSONDecodeError):
                    previous = {}
            now = datetime.now().astimezone().isoformat()
            record = {
                "schema_version": "tactile_intervals_v1",
                "rollout_id": rollout_id,
                "review_status": "complete" if normalized else "unreviewed",
                "created_at": previous.get("created_at", now),
                "updated_at": now,
                "tactile_intervals": normalized,
            }
            self._atomic_text(target, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            return normalized

    def rollout_payload(self) -> dict[str, Any]:
        annotations = self.annotations_by_rollout()
        rows = []
        for original in self.load_rollouts():
            row = dict(original)
            events = annotations.get(str(row.get("id")), [])
            row["annotation_events"] = events
            row["annotation_status"] = "complete" if events else "unreviewed"
            rows.append(row)
        source = self.annotation_seed_path()
        return {
            "rollouts": rows,
            "manifests": [
                {
                    "path": self._display_path(self.manifest_path, self.source_root),
                    "label": "failrecovery",
                    "rollouts": len(rows),
                    "valid": self.manifest_path.is_file(),
                }
            ],
            "annotation_source": self._display_path(source, self.source_root) if source else None,
            "annotation_target": str(self.annotation_target_path().relative_to(self.root)),
            "event_types": EVENT_TYPES,
        }

    # ---------- analysis / run browser ----------

    def analysis_summary(self) -> dict[str, Any]:
        rollouts = self.load_rollouts()
        grouped = self.annotations_by_rollout()
        counter: Counter[int] = Counter()
        duration_by_key: dict[int, list[int]] = defaultdict(list)
        task_rows: dict[str, dict[str, Any]] = {}
        for record in rollouts:
            rollout_id = str(record.get("id") or "")
            task = str(record.get("task_key") or record.get("task_description") or record.get("task_id") or "unknown")
            task_row = task_rows.setdefault(task, {"task": task, "rollouts": 0, "reviewed": 0, "events": 0})
            task_row["rollouts"] += 1
            events = grouped.get(rollout_id, [])
            if events:
                task_row["reviewed"] += 1
            task_row["events"] += len(events)
            for event in events:
                key = int(event["event_key"])
                counter[key] += 1
                duration_by_key[key].append(int(event["end_frame"]) - int(event["start_frame"]) + 1)
        type_rows = []
        for key, spec in EVENT_TYPES.items():
            durations = duration_by_key[key]
            type_rows.append(
                {
                    "event_key": key,
                    **spec,
                    "count": counter[key],
                    "mean_duration_frames": (sum(durations) / len(durations)) if durations else None,
                }
            )
        reviewed = sum(1 for record in rollouts if grouped.get(str(record.get("id") or "")))
        return {
            "rollouts": len(rollouts),
            "reviewed_rollouts": reviewed,
            "coverage": reviewed / len(rollouts) if rollouts else 0.0,
            "events": sum(counter.values()),
            "event_types": type_rows,
            "tasks": sorted(task_rows.values(), key=lambda row: row["task"]),
            "annotation_source": self._display_path(self.annotation_seed_path(), self.source_root) if self.annotation_seed_path() else None,
            "annotation_target": str(self.annotation_target_path().relative_to(self.root)),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        root = self.project_file(self._settings["runs_root"])
        if not root.is_dir():
            return []
        markers = {
            "README.md",
            "results.json",
            "suite_status.json",
            "training_manifest.json",
            "data_manifest.json",
            "dataset_manifest.json",
        }
        found: dict[Path, set[str]] = defaultdict(set)
        for marker in markers:
            for path in root.glob(f"**/{marker}"):
                if not path.is_file():
                    continue
                relative = path.parent.relative_to(root)
                if len(relative.parts) > 4:
                    continue
                found[path.parent].add(marker)
        rows: list[dict[str, Any]] = []
        for directory, names in found.items():
            stat = directory.stat()
            rows.append(
                {
                    "path": str(directory.relative_to(self.root)),
                    "name": directory.name,
                    "markers": sorted(names),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                }
            )
        rows.sort(key=lambda row: row["modified_at"], reverse=True)
        return rows[:200]


class TactileHandler(BaseHTTPRequestHandler):
    app: TactileApplication
    protocol_version = "HTTP/1.0"
    server_version = "TactileWebUI/2.0"

    CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

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

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0 or length > 8 * 1024 * 1024:
            raise ValueError("JSON request body is empty or too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

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
    def _camera_for_rollout(rollout: dict[str, Any], query: dict[str, list[str]], default: str) -> str:
        camera_paths = rollout.get("camera_video_paths")
        if not isinstance(camera_paths, dict) or not camera_paths:
            raise KeyError("Rollout has no camera videos")
        camera = str(query.get("camera", [default])[0] or default).strip()
        if camera in camera_paths:
            return camera
        observation = rollout.get("observation_key")
        if isinstance(observation, str) and observation in camera_paths:
            return observation
        for preferred in (default, "cam_high", "cam_wrist", "cam_left_wrist", "cam_right_wrist"):
            if preferred in camera_paths:
                return preferred
        return str(next(iter(camera_paths)))

    def _route_tactile(self, rollout_id: str, resource: str, query: dict[str, list[str]]) -> None:
        camera = str(query.get("camera", [self.app.settings["default_camera"]])[0] or self.app.settings["default_camera"])
        if resource == "series":
            self.json_response(HTTPStatus.OK, {"tactile": self.app.tactile.series(rollout_id, camera)})
            return
        if resource == "frame":
            frame = int(query.get("frame", ["0"])[0])
            self.json_response(HTTPStatus.OK, {"tactile": self.app.tactile.frame(rollout_id, camera, frame)})
            return
        if resource == "sprite":
            frame = int(query.get("frame", ["0"])[0])
            kind = str(query.get("kind", [self.app.settings["default_tactile_kind"]])[0] or self.app.settings["default_tactile_kind"])
            self.serve_png(self.app.tactile.sprite(rollout_id, camera, frame, kind))
            return
        if resource == "image":
            finger = str(query.get("finger", [""])[0] or "")
            event_id = str(query.get("event_id", [""])[0] or "")
            kind = str(query.get("kind", [self.app.settings["default_tactile_kind"]])[0] or self.app.settings["default_tactile_kind"])
            self.serve_png(self.app.tactile.image(rollout_id, finger, event_id, kind))
            return
        self.json_error(HTTPStatus.NOT_FOUND, "Tactile resource not found")

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
                        "manifest_exists": self.app.manifest_path.is_file(),
                        "rollouts": len(self.app.load_rollouts()),
                        "annotations": str(self.app.annotation_target_path()),
                    },
                )
                return
            if path == "/api/rollouts":
                self.json_response(HTTPStatus.OK, self.app.rollout_payload())
                return
            if path == "/api/event-types":
                self.json_response(HTTPStatus.OK, {"event_types": EVENT_TYPES})
                return
            if path == "/api/analysis":
                self.json_response(HTTPStatus.OK, self.app.analysis_summary())
                return
            if path == "/api/runs":
                self.json_response(HTTPStatus.OK, {"runs": self.app.list_runs(), "runs_root": self.app.settings["runs_root"]})
                return
            if path == "/api/settings":
                source = self.app.annotation_seed_path()
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "settings": self.app.settings,
                        "manifest_path": self.app._display_path(self.app.manifest_path, self.app.source_root),
                        "source_project_root_resolved": str(self.app.source_root),
                        "manifest_exists": self.app.manifest_path.is_file(),
                        "annotation_source": self.app._display_path(source, self.app.source_root) if source else None,
                        "annotation_target": str(self.app.annotation_target_path().relative_to(self.app.root)),
                    },
                )
                return
            if path.startswith("/api/annotations/"):
                rollout_id = path[len("/api/annotations/"):].strip("/")
                if rollout_id not in self.app.rollout_map():
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                events = self.app.annotations_by_rollout().get(rollout_id, [])
                self.json_response(HTTPStatus.OK, {"rollout_id": rollout_id, "events": events})
                return
            if path.startswith("/api/videos/"):
                rollout_id = path[len("/api/videos/"):].strip("/")
                rollout = self.app.rollout_map().get(rollout_id)
                if not rollout:
                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")
                    return
                camera = self._camera_for_rollout(rollout, query, self.app.settings["default_camera"])
                camera_paths = rollout.get("camera_video_paths") or {}
                self.serve_video(self.app.source_file(camera_paths.get(camera), ".mp4"))
                return
            if path.startswith("/api/tactile/"):
                parts = path.strip("/").split("/")
                if len(parts) != 4 or parts[:2] != ["api", "tactile"]:
                    self.json_error(HTTPStatus.NOT_FOUND, "Tactile resource not found")
                    return
                self._route_tactile(parts[2], parts[3], query)
                return
            if path in {"", "/"}:
                self.serve_static(self.app.static_root / "index.html")
                return
            if path in {"/tactile", "/tactile/"}:
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

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            payload = self._read_json_body()
            if path == "/api/settings":
                self.json_response(HTTPStatus.OK, {"settings": self.app.save_settings(payload)})
                return
            if path.startswith("/api/annotations/"):
                rollout_id = path[len("/api/annotations/"):].strip("/")
                events = self.app.save_rollout_annotations(rollout_id, payload.get("events"))
                self.json_response(
                    HTTPStatus.OK,
                    {
                        "rollout_id": rollout_id,
                        "events": events,
                        "annotation_target": str(self.app.annotation_target_path().relative_to(self.app.root)),
                    },
                )
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
    print(f"Tactile WebUI: http://{args.host}:{args.port}/", flush=True)
    print(f"Dataset manifest: {app.manifest_path}", flush=True)
    print(f"Annotation target: {app.annotation_target_path()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
