"""Project-local annotation label registry; display metadata never changes event IDs."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_LABELS: list[dict[str, Any]] = [{"id":1,"name":"Success","description":"Successful local action interval; not necessarily the rollout outcome.","color":"#4f7df3","scope":"all","active":True},{"id":2,"name":"Failure","description":"Failed local action interval; may occur in an eventually successful rollout.","color":"#d96c6c","scope":"all","active":True},{"id":3,"name":"Dropped object","description":"The object is dropped during manipulation.","color":"#8a6fd1","scope":"all","active":True},{"id":4,"name":"Wrong object","description":"The robot interacts with the wrong object.","color":"#2f9e8b","scope":"all","active":True}]
COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class LabelRegistry:
    """Labels 1-4 are stable; new IDs are append-only and scopes are immutable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        default = {"schema_version": 1, "dataset": "failrecovery", "labels": DEFAULT_LABELS}
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
        # The former failure-only restriction on IDs 3/4 came from a
        # historical annotation-cleanup rule, not from manual annotation
        # semantics. Upgrade legacy local registries without dropping any
        # custom label names, colors, or extra IDs. A later Settings save will
        # persist the updated scopes.
        if isinstance(loaded, dict) and isinstance(loaded.get("labels"), list):
            for row in loaded["labels"]:
                if isinstance(row, dict) and row.get("id") in (3, 4) and row.get("scope") == "failure":
                    row["scope"] = "all"
        self._data = self._validate(loaded)

    @staticmethod
    def _validate(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("annotation labels require schema_version=1")
        if payload.get("dataset") != "failrecovery":
            raise ValueError("annotation labels dataset must be failrecovery")
        labels = payload.get("labels")
        if not isinstance(labels, list) or not 4 <= len(labels) <= 64:
            raise ValueError("annotation labels must contain 4-64 entries")
        seen: set[int] = set()
        normalized = []
        for raw in labels:
            if not isinstance(raw, dict):
                raise ValueError("each annotation label must be an object")
            label_id = raw.get("id")
            if type(label_id) is not int or not 1 <= label_id <= 9999 or label_id in seen:
                raise ValueError(f"invalid or duplicated label ID: {label_id!r}")
            seen.add(label_id)
            name = raw.get("name")
            description = raw.get("description")
            color = raw.get("color")
            scope = raw.get("scope")
            active = raw.get("active")
            if not isinstance(name, str) or not 1 <= len(name.strip()) <= 64:
                raise ValueError(f"label {label_id}: name must be 1-64 characters")
            if not isinstance(description, str) or len(description) > 500:
                raise ValueError(f"label {label_id}: description must be <=500 characters")
            if not isinstance(color, str) or not COLOR.fullmatch(color):
                raise ValueError(f"label {label_id}: color must be #RRGGBB")
            if scope not in ("all", "failure"):
                raise ValueError(f"label {label_id}: scope must be all or failure")
            if type(active) is not bool:
                raise ValueError(f"label {label_id}: active must be boolean")
            normalized.append({
                "id": label_id, "name": name.strip(), "description": description.strip(),
                "color": color.lower(), "scope": scope, "active": active,
            })
        by_id = {row["id"]: row for row in normalized}
        for default in DEFAULT_LABELS:
            if default["id"] not in by_id or by_id[default["id"]]["scope"] != default["scope"]:
                raise ValueError(f"label {default['id']}: required ID/scope cannot change")
        return {"schema_version": 1, "dataset": "failrecovery", "labels": sorted(normalized, key=lambda row: row["id"])}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def get(self, label_id: int) -> dict[str, Any] | None:
        with self._lock:
            for row in self._data["labels"]:
                if row["id"] == label_id:
                    return dict(row)
        return None

    def update(self, payload: Any) -> dict[str, Any]:
        validated = self._validate(payload)
        with self._lock:
            previous = {row["id"]: row for row in self._data["labels"]}
            current = {row["id"]: row for row in validated["labels"]}
            for label_id, original in previous.items():
                if label_id not in current:
                    raise ValueError(f"label {label_id}: existing IDs cannot be removed or reused; deactivate instead")
                if current[label_id]["scope"] != original["scope"]:
                    raise ValueError(f"label {label_id}: existing scope cannot change")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(validated, indent=2, ensure_ascii=False) + "\n"
            fd, temporary = tempfile.mkstemp(prefix=".labels-", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(content)
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self._data = validated
            return deepcopy(self._data)
