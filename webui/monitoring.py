from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any


class WebUIMonitor:
    """Low-overhead structured monitoring for the standalone WebUI.

    All runtime logs live below ``logs/webui``. High-frequency successful
    tactile image requests stay in the bounded in-memory window but are sampled
    on disk so monitoring cannot become the performance problem it measures.
    """

    def __init__(self, root: Path, recent_limit: int = 600) -> None:
        self.root = root.resolve()
        self.server_dir = self.root / "server"
        self.client_dir = self.root / "client"
        self.server_dir.mkdir(parents=True, exist_ok=True)
        self.client_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_limit)
        self._client_recent: deque[dict[str, Any]] = deque(maxlen=recent_limit)
        self._started_wall = datetime.now().astimezone()
        self._started_mono = time.monotonic()
        self._request_sequence = 0

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat()

    @staticmethod
    def _daily_path(directory: Path) -> Path:
        return directory / f"{datetime.now().astimezone().date().isoformat()}.jsonl"

    def _append_many(self, directory: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self._daily_path(directory)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)

    def _append(self, directory: Path, payload: dict[str, Any]) -> None:
        self._append_many(directory, [payload])

    def record_request(self, **payload: Any) -> None:
        row = {"ts": self._now(), "kind": "http", **payload}
        status = int(row.get("status") or 0)
        duration = float(row.get("duration_ms") or 0.0)
        resource = str(row.get("resource") or "other")
        path = str(row.get("path") or "")
        with self._lock:
            self._request_sequence += 1
            request_sequence = self._request_sequence
            self._recent.append(row)
            persist = (
                status >= 400
                or bool(row.get("client_disconnected"))
                or resource == "video"
                or (resource == "api" and path != "/api/telemetry")
                or duration >= 250.0
                or (resource == "tactile" and request_sequence % 50 == 0)
            )
        # Never hold the shared monitor lock across filesystem I/O. Under image
        # load bursts that lock used to serialize otherwise independent request
        # threads and could make the monitor itself visible in latency traces.
        if persist:
            self._append(self.server_dir, row)

    def record_server_event(self, event: str, **payload: Any) -> None:
        row = {"ts": self._now(), "kind": "server", "event": event, **payload}
        with self._lock:
            self._recent.append(row)
        self._append(self.server_dir, row)

    def record_client_events(self, events: list[dict[str, Any]]) -> int:
        rows: list[dict[str, Any]] = []
        persisted: list[dict[str, Any]] = []
        for raw in events[:200]:
            if not isinstance(raw, dict):
                continue
            row = {"ts_server": self._now(), "kind": "client", **raw}
            rows.append(row)
            event = str(row.get("event") or "")
            level = str(row.get("level") or "info")
            # Successful high-rate tactile resource timing is sampled on the
            # client before upload. Persist every uploaded diagnostic event.
            if level == "error" or event or row.get("duration_ms") is not None:
                persisted.append(row)
        with self._lock:
            self._client_recent.extend(rows)
        # One append/open per telemetry batch instead of one per event.
        self._append_many(self.client_dir, persisted)
        return len(rows)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return round(ordered[index], 3)

    def _latency(self, rows: list[dict[str, Any]]) -> dict[str, float | None]:
        values = [float(row["duration_ms"]) for row in rows if row.get("duration_ms") is not None]
        return {
            "p50": self._percentile(values, 0.50),
            "p95": self._percentile(values, 0.95),
            "max": round(max(values), 3) if values else None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            recent = list(self._recent)
            client_recent = list(self._client_recent)
        requests = [row for row in recent if row.get("kind") == "http"]
        statuses = Counter(str(row.get("status", "unknown")) for row in requests)
        resources = Counter(str(row.get("resource", "other")) for row in requests)
        errors = [row for row in requests if int(row.get("status") or 0) >= 400 or row.get("client_disconnected")]
        client_errors = [
            row for row in client_recent
            if row.get("level") == "error" or row.get("event") in {"error", "stalled", "watchdog_timeout", "network_error", "tactile_image_error"}
        ]
        by_resource = {
            resource: self._latency([row for row in requests if row.get("resource") == resource])
            for resource in ("video", "api", "tactile", "static")
        }
        return {
            "started_at": self._started_wall.isoformat(),
            "uptime_seconds": round(time.monotonic() - self._started_mono, 3),
            "requests": len(requests),
            "status_counts": dict(statuses),
            "resource_counts": dict(resources),
            "latency_ms": self._latency(requests),
            "latency_by_resource_ms": by_resource,
            "server_errors": errors[-12:],
            "client_events": client_recent[-20:],
            "client_errors": client_errors[-12:],
            "log_paths": {
                "server": str(self.server_dir),
                "client": str(self.client_dir),
            },
        }
