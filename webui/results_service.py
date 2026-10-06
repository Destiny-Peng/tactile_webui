from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class OnlineResultsService:
    """Read online-detection prediction CSVs produced by SHARPA experiments."""

    REQUIRED = {"rollout_id", "label"}

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def _runs_root(self, value: str) -> Path:
        root = (self.project_root / value).resolve()
        root.relative_to(self.project_root)
        return root

    @staticmethod
    def _probability_columns(fieldnames: list[str]) -> list[str]:
        names = set(fieldnames)
        preferred = ["p_in_progress", "p_success", "p_failure"]
        if all(name in names for name in preferred):
            return preferred
        for candidate in (["p0", "p1", "p2"], ["p_0", "p_1", "p_2"]):
            if all(name in names for name in candidate):
                return list(candidate)
        return []

    def list_sources(self, runs_root_value: str) -> list[dict[str, Any]]:
        root = self._runs_root(runs_root_value)
        if not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in root.glob("**/test_predictions.csv"):
            if not path.is_file():
                continue
            try:
                relative_to_runs = path.relative_to(root)
            except ValueError:
                continue
            if len(relative_to_runs.parts) > 9:
                continue
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fields = list(reader.fieldnames or [])
                    first = next(reader, None)
            except (OSError, csv.Error):
                continue
            if not self.REQUIRED.issubset(fields):
                continue
            probs = self._probability_columns(fields)
            if not probs:
                continue
            relative = path.relative_to(self.project_root).as_posix()
            metrics_path = path.parent / "metrics.json"
            metrics = None
            if metrics_path.is_file():
                try:
                    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metrics = loaded
                except (OSError, json.JSONDecodeError):
                    metrics = None
            rows.append(
                {
                    "id": relative,
                    "path": relative,
                    "name": path.parent.relative_to(root).as_posix(),
                    "probability_columns": probs,
                    "has_prediction": "prediction" in fields,
                    "preview_rollout": first.get("rollout_id") if first else None,
                    "metrics": metrics,
                    "modified_at": path.stat().st_mtime,
                }
            )
        rows.sort(key=lambda row: (row["modified_at"], row["path"]), reverse=True)
        return rows

    def _resolve_source(self, runs_root_value: str, source: str) -> Path:
        root = self._runs_root(runs_root_value)
        raw = Path(source)
        path = raw.resolve() if raw.is_absolute() else (self.project_root / raw).resolve()
        path.relative_to(root)
        if path.name != "test_predictions.csv" or not path.is_file():
            raise ValueError("result source must be an existing test_predictions.csv below runs_root")
        return path

    @staticmethod
    def _frame_from_row(row: dict[str, str]) -> int:
        for key in ("video_frame", "frame"):
            value = row.get(key)
            if value not in (None, ""):
                return int(float(value))
        sample_frames = row.get("sample_frames")
        if sample_frames:
            parsed = json.loads(sample_frames)
            if isinstance(parsed, list) and parsed:
                return int(parsed[-1])
        step = row.get("step")
        if step not in (None, ""):
            return int(float(step))
        raise ValueError("prediction row has no frame/sample_frames/step coordinate")

    def curve(self, runs_root_value: str, source: str, rollout_id: str) -> dict[str, Any]:
        path = self._resolve_source(runs_root_value, source)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            probability_columns = self._probability_columns(fields)
            if not probability_columns:
                raise ValueError("prediction CSV has no supported three-class probability columns")
            points: list[dict[str, Any]] = []
            for row in reader:
                if str(row.get("rollout_id") or "") != rollout_id:
                    continue
                try:
                    probabilities = [float(row[name]) for name in probability_columns]
                    label = int(float(row["label"]))
                    prediction = int(float(row["prediction"])) if row.get("prediction") not in (None, "") else max(range(len(probabilities)), key=probabilities.__getitem__)
                    frame = self._frame_from_row(row)
                except (ValueError, TypeError, json.JSONDecodeError, KeyError):
                    continue
                points.append(
                    {
                        "frame": frame,
                        "label": label,
                        "prediction": prediction,
                        "probabilities": probabilities,
                    }
                )
        points.sort(key=lambda row: row["frame"])
        return {
            "source": path.relative_to(self.project_root).as_posix(),
            "rollout_id": rollout_id,
            "probability_labels": [0, 1, 2],
            "points": points,
        }
