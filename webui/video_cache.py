"""Cache mapping and batch H.264 job orchestration for original recorder videos.

All writes stay under the standalone repository's cache/ and logs/ trees.
The input videos and manifests are never modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cache_path(source: Path, cache_dir: Path) -> Path:
    source = source.expanduser().resolve()
    info = source.stat()
    identity = f"{source}\0{info.st_size}\0{info.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.h264.mp4"


def compatible_video(source: Path, cache_dir: Path) -> Path:
    try:
        output = cache_path(source, cache_dir)
        if output.is_file() and output.stat().st_size > 0:
            return output
    except OSError:
        pass
    return source


class VideoTranscodeManager:
    """One batch at a time; durable job progress and stderr/stdout log."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.cache_dir = self.root / "cache/videos/h264"
        self.jobs_root = self.root / "logs/webui/transcode"
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None

    def _latest_directory(self) -> Path | None:
        if not self.jobs_root.is_dir():
            return None
        entries = sorted(
            (x for x in self.jobs_root.iterdir() if x.is_dir() and (x / "state.json").is_file()),
            key=lambda x: x.name, reverse=True,
        )
        return entries[0] if entries else None

    def snapshot(self) -> dict[str, Any]:
        folder = self._latest_directory()
        if folder is None:
            return {"job": None}
        try:
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {"status": "unknown"}
        with self.lock:
            process = self.process
            if process is not None and process.poll() is not None:
                self.process = None
                if state.get("status") in {"running", "queued"}:
                    state["status"] = "failed"
                    state["error"] = f"transcode process exited {process.returncode} without final status"
            elif process is None and state.get("status") in {"queued", "running"}:
                # The process was detached by a WebUI restart. Progress file
                # is updated by the child; show its recorded status as-is.
                pid = state.get("pid")
                if pid:
                    try:
                        os.kill(int(pid), 0)
                    except (OSError, ValueError):
                        state["status"] = "interrupted"
                else:
                    state["status"] = "interrupted"
        log_file = folder / "run.log"
        if log_file.is_file():
            with log_file.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 7000))
                tail = stream.read().decode("utf-8", errors="replace")
        else:
            tail = ""
        return {"job": {"job_id": folder.name, **state}, "log_tail": tail}

    def submit(self, app: Any, manifests: list[str]) -> dict[str, Any]:
        if not isinstance(manifests, list) or not manifests or len(manifests) > 64:
            raise ValueError("select 1–64 manifest entries")
        if not all(isinstance(x, str) and x for x in manifests):
            raise ValueError("manifest paths must be non-empty strings")
        app.load_rollouts()
        accepted = {info["path"] for info in app._manifest_info if info["valid"]}
        selection = set(manifests)
        if len(selection) != len(manifests) or not selection.issubset(accepted):
            raise ValueError("unknown, invalid or duplicated manifest selection")
        unique = set()
        paths: list[str] = []
        for row in app.load_rollouts():
            if row.get("manifest_source") not in selection:
                continue
            camera_paths = row.get("camera_video_paths") or {}
            for name, raw in camera_paths.items():
                path = app.source_file(raw, ".mp4", str(row["id"]))
                if path not in unique:
                    unique.add(path)
                    paths.append(str(path))
        if not paths:
            raise ValueError("selected manifests contain no video paths")
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("another H.264 batch is currently running")
            last = self._latest_directory()
            if last is not None:
                try:
                    earlier = json.loads((last / "state.json").read_text(encoding="utf-8"))
                    pid = earlier.get("pid")
                    if earlier.get("status") in {"running", "queued"} and pid:
                        os.kill(int(pid), 0)
                        raise ValueError("an H.264 batch is still running from a previous WebUI session")
                except ProcessLookupError:
                    pass
                except (OSError, ValueError) as exc:
                    if isinstance(exc, ValueError):
                        raise
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            name = f"{stamp}-{uuid.uuid4().hex[:8]}"
            folder = self.jobs_root / name
            folder.mkdir(parents=True, exist_ok=False)
            plan = folder / "plan.json"
            state = folder / "state.json"
            plan.write_text(json.dumps({"videos": paths, "manifests": manifests}, indent=2) + "\n")
            state.write_text(json.dumps({
                "status": "queued", "total": len(paths), "processed": 0,
                "manifests": manifests,
            }, indent=2) + "\n")
            command = [
                sys.executable, "-u", str(self.root / "tools/transcode_manifest_videos_h264.py"),
                "--plan", str(plan), "--state", str(state),
                "--cache-dir", str(self.cache_dir),
            ]
            with (folder / "run.log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.Popen(
                        command, cwd=str(self.root), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError:
                    state.write_text(json.dumps({"status": "failed", "error": "unable to launch batch process"}))
                    raise
            self.process = process
        return {"job_id": name, "status": "queued", "total": len(paths),
                "manifests": manifests}
