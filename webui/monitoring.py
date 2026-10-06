from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any


class WebUIMonitor:
    """Small structured monitor for the standalone WebUI.

    Persistent logs are intentionally namespaced below logs/webui so they do
    not mix with experiment/training logs. A bounded in-memory window powers
    the Diagnostics panel without rescanning log files.
    """

    def __init__(self, root: Path, recent_limit: int = 400) -> None:
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

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat()

    @staticmethod
    def _daily_path(directory: Path) -> Path:
        return directory / f"{datetime.now().astimezone().date().isoformat()}.jsonl"

    def _append(self, directory: Path, payload: dict[str, Any]) -> None:
        path = self._daily_path(directory)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def record_request(self, **payload: Any) -> None:
        row = {"ts": self._now(), "kind": "http", **payload}
        with self._lock:
            self._recent.append(row)
            self._append(self.server_dir, row)

    def record_server_event(self, event: str, **payload: Any) -> None:
        row = {"ts": self._now(), "kind": "server", "event": event, **payload}
        with self._lock:
            self._recent.append(row)
            self._append(self.server_dir, row)

    def record_client_events(self, events: list[dict[str, Any]]) -> int:
        accepted = 0
        with self._lock:
            for raw in events[:200]:
                if not isinstance(raw, dict):
                    continue
                row = {"ts_server": self._now(), "kind": "client", **raw}
                self._client_recent.append(row)
                self._append(self.client_dir, row)
                accepted += 1
        return accepted

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            recent = list(self._recent)
            client_recent = list(self._client_recent)
        requests = [row for row in recent if row.get("kind") == "http"]
        durations = [float(row["duration_ms"]) for row in requests if row.get("duration_ms") is not None]
        statuses = Counter(str(row.get("status", "unknown")) for row in requests)
        resources = Counter(str(row.get("resource", "other")) for row in requests)
        errors = [row for row in requests if int(row.get("status") or 0) >= 400]
        client_errors = [
            row for row in client_recent
            if row.get("level") == "error" or row.get("event") in {"error", "stalled", "watchdog_timeout", "network_error"}
        ]
        return {
            "started_at": self._started_wall.isoformat(),
            "uptime_seconds": round(time.monotonic() - self._started_mono, 3),
            "requests": len(requests),
            "status_counts": dict(statuses),
            "resource_counts": dict(resources),
            "latency_ms": {
                "p50": self._percentile(durations, 0.50),
                "p95": self._percentile(durations, 0.95),
                "max": max(durations) if durations else None,
            },
            "server_errors": errors[-12:],
            "client_events": client_recent[-20:],
            "client_errors": client_errors[-12:],
            "log_paths": {
                "server": str(self.server_dir),
                "client": str(self.client_dir),
            },
        }
