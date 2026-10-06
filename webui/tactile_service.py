"""FailRecovery tactile reader used only by the LF3R Results WebUI."""

from __future__ import annotations

import bisect
import json
import struct
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
KINDS = ("raw", "deform")


@dataclass
class EpisodeTactile:
    rollout_id: str
    frames: list[dict[str, Any]]
    camera_frames: dict[str, list[int]]
    camera_rows: dict[str, list[int]]
    events: dict[str, dict[str, Any]]
    finger_events: dict[tuple[str, str], dict[str, Any]]
    streams: dict[str, dict[str, Path]]


class FailRecoveryTactileService:
    """Small, dataset-specific index for failrecovery_manifest.jsonl."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.manifest_path = (
            self.project_root
            / "datasets"
            / "lf3r_failure_rollouts"
            / "v1"
            / "failrecovery_manifest.jsonl"
        )
        self.episodes: dict[str, EpisodeTactile] = {}
        self._series_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._sprite_cache: OrderedDict[tuple[str, str, int, str], bytes] = OrderedDict()
        self._sprite_cache_lock = threading.Lock()
        self._sprite_cache_limit = 96
        self._manifest_rows: dict[str, dict[str, Any]] = {}
        self._episode_locks: dict[str, threading.Lock] = {}
        self._load_manifest()

    def _project_file(self, value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing project-relative tactile path")
        path = (self.project_root / value).resolve()
        try:
            path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("tactile path escapes project root") from exc
        return path

    @staticmethod
    def _jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def _stream_paths(self, row: dict[str, Any]) -> dict[str, dict[str, Path]]:
        raw = row.get("tactile_stream_paths")
        if not isinstance(raw, dict):
            return {}
        result: dict[str, dict[str, Path]] = {}
        for finger in FINGERS:
            entry = raw.get(finger)
            finger_paths: dict[str, Path] = {}
            if isinstance(entry, dict):
                for kind in KINDS:
                    value = (
                        entry.get(kind)
                        or entry.get(kind + "_path")
                        or entry.get(kind + "_u8")
                    )
                    if value:
                        finger_paths[kind] = self._project_file(value)
            for kind in KINDS:
                if kind in finger_paths:
                    continue
                value = (
                    raw.get(finger + "_" + kind)
                    or raw.get(finger + "." + kind)
                    or raw.get(finger + "_" + kind + "_path")
                )
                if value:
                    finger_paths[kind] = self._project_file(value)
            if finger_paths:
                result[finger] = finger_paths
        return result

    def _load_manifest(self) -> None:
        """Discover episodes without reading their frame/event indices."""
        if not self.manifest_path.is_file():
            return
        try:
            manifest_rows = self._jsonl(self.manifest_path)
        except (OSError, json.JSONDecodeError):
            return
        for row in manifest_rows:
            rollout_id = str(row.get("id") or "").strip()
            if rollout_id:
                self._manifest_rows[rollout_id] = row
                self._episode_locks[rollout_id] = threading.Lock()

    def _load_episode(self, rollout_id: str, row: dict[str, Any]) -> EpisodeTactile:
        frames_path = self._project_file(row.get("synchronized_frames_path"))
        events_path = self._project_file(row.get("tactile_events_path"))
        streams = self._stream_paths(row)
        frames = self._jsonl(frames_path)
        event_rows = self._jsonl(events_path)

        events: dict[str, dict[str, Any]] = {}
        finger_events: dict[tuple[str, str], dict[str, Any]] = {}
        for event in event_rows:
            event_id = event.get("event_id")
            if event_id is None:
                continue
            key = str(event_id)
            events[key] = event
            finger = str(event.get("finger") or event.get("finger_name") or "").strip().lower()
            if finger:
                finger_events[(finger, key)] = event

        camera_pairs: dict[str, list[tuple[int, int]]] = {}
        for row_index, frame_row in enumerate(frames):
            indices = frame_row.get("camera_frame_indices")
            if not isinstance(indices, dict):
                continue
            for camera, value in indices.items():
                try:
                    frame_number = int(value)
                except (TypeError, ValueError):
                    continue
                camera_pairs.setdefault(str(camera), []).append((frame_number, row_index))

        camera_frames: dict[str, list[int]] = {}
        camera_rows: dict[str, list[int]] = {}
        for camera, pairs in camera_pairs.items():
            pairs.sort(key=lambda item: item[0])
            camera_frames[camera] = [item[0] for item in pairs]
            camera_rows[camera] = [item[1] for item in pairs]

        return EpisodeTactile(
            rollout_id=rollout_id,
            frames=frames,
            camera_frames=camera_frames,
            camera_rows=camera_rows,
            events=events,
            finger_events=finger_events,
            streams=streams,
        )

    def _episode(self, rollout_id: str) -> EpisodeTactile:
        row = self._manifest_rows.get(rollout_id)
        if row is None:
            raise KeyError("tactile rollout not found")
        # Separate locks let unrelated episodes load concurrently. Publish an
        # index only when complete, and reuse it across all tactile endpoints.
        with self._episode_locks[rollout_id]:
            episode = self.episodes.get(rollout_id)
            if episode is None:
                try:
                    episode = self._load_episode(rollout_id, row)
                except (OSError, ValueError) as exc:
                    raise KeyError("tactile rollout not found") from exc
                self.episodes[rollout_id] = episode
            return episode

    def has_rollout(self, rollout_id: str) -> bool:
        return rollout_id in self._manifest_rows

    @staticmethod
    def _nearest_index(values: list[int], requested: int) -> int:
        if not values:
            raise KeyError("camera has no synchronized frames")
        pos = bisect.bisect_left(values, requested)
        if pos <= 0:
            return 0
        if pos >= len(values):
            return len(values) - 1
        before = values[pos - 1]
        after = values[pos]
        return pos - 1 if abs(requested - before) <= abs(after - requested) else pos

    def _event(self, episode: EpisodeTactile, finger: str, event_id: Any) -> dict[str, Any] | None:
        if event_id is None:
            return None
        key = str(event_id)
        return episode.finger_events.get((finger, key)) or episode.events.get(key)

    def frame(self, rollout_id: str, camera: str, video_frame: int) -> dict[str, Any]:
        episode = self._episode(rollout_id)
        frames = episode.camera_frames.get(camera)
        rows = episode.camera_rows.get(camera)
        if not frames or not rows:
            raise KeyError("camera has no tactile synchronization")

        index = self._nearest_index(frames, int(video_frame))
        row_index = rows[index]
        row = episode.frames[row_index]
        tactile = row.get("tactile") if isinstance(row.get("tactile"), dict) else {}

        fingers: dict[str, Any] = {}
        for finger in FINGERS:
            sync = tactile.get(finger) if isinstance(tactile.get(finger), dict) else {}
            event_id = sync.get("event_id")
            event = self._event(episode, finger, event_id)
            if event is None and event_id is None:
                continue
            event = event or {}
            sync_valid = sync.get("valid")
            fingers[finger] = {
                "event_id": event_id,
                "sensor_ts_ns": event.get("sensor_ts_ns"),
                "receive_wall_ns": event.get("receive_wall_ns"),
                "receive_mono_ns": event.get("receive_mono_ns"),
                "f6": event.get("f6"),
                "valid": event.get("valid") if sync_valid is None else sync_valid,
                "event_valid": event.get("valid"),
                "stale": sync.get("stale"),
                "age_ms": sync.get("age_ms"),
                "image_kinds": [
                    kind for kind in KINDS
                    if kind in episode.streams.get(finger, {})
                    and event.get(kind + "_offset_bytes") is not None
                    and event.get(kind + "_length_bytes") is not None
                    and event.get(kind + "_shape") is not None
                ],
            }

        return {
            "available": True,
            "rollout_id": rollout_id,
            "camera": camera,
            "requested_video_frame": int(video_frame),
            "matched_video_frame": frames[index],
            "sync_row": row_index,
            "complete": row.get("complete"),
            "elapsed_s": row.get("elapsed_s"),
            "tick_wall_ns": row.get("tick_wall_ns"),
            "tick_mono_ns": row.get("tick_mono_ns"),
            "fingers": fingers,
        }

    def series(self, rollout_id: str, camera: str) -> dict[str, Any]:
        cache_key = (rollout_id, camera)
        cached = self._series_cache.get(cache_key)
        if cached is not None:
            return cached

        episode = self._episode(rollout_id)
        frames = episode.camera_frames.get(camera)
        rows = episode.camera_rows.get(camera)
        if not frames or not rows:
            raise KeyError("camera has no tactile synchronization")

        finger_series: dict[str, list[dict[str, Any]]] = {
            finger: [] for finger in FINGERS
        }
        sync_frames: list[dict[str, Any]] = []
        last_event_id: dict[str, str | None] = {
            finger: None for finger in FINGERS
        }
        for video_frame, row_index in zip(frames, rows):
            row = episode.frames[row_index]
            tactile = (
                row.get("tactile")
                if isinstance(row.get("tactile"), dict)
                else {}
            )
            sync_frame = {
                "frame": int(video_frame),
                "sync_row": int(row_index),
                "complete": row.get("complete"),
                "fingers": {},
            }
            for finger in FINGERS:
                sync = (
                    tactile.get(finger)
                    if isinstance(tactile.get(finger), dict)
                    else {}
                )
                event_id = sync.get("event_id")
                sync_frame["fingers"][finger] = {
                    "event_id": event_id,
                    "valid": sync.get("valid"),
                    "stale": sync.get("stale"),
                    "age_ms": sync.get("age_ms"),
                }
                event_key = None if event_id is None else str(event_id)
                if event_key is None or event_key == last_event_id[finger]:
                    continue
                last_event_id[finger] = event_key
                event = self._event(episode, finger, event_id)
                if event is None:
                    continue
                f6 = event.get("f6")
                if not isinstance(f6, list) or not f6:
                    continue
                try:
                    values = [float(value) for value in f6]
                except (TypeError, ValueError):
                    continue
                sync_valid = sync.get("valid")
                finger_series[finger].append(
                    {
                        "frame": int(video_frame),
                        "event_id": event_id,
                        "f6": values,
                        "valid": (
                            event.get("valid")
                            if sync_valid is None
                            else sync_valid
                        ),
                        "stale": sync.get("stale"),
                        "age_ms": sync.get("age_ms"),
                        "image_kinds": [
                            kind for kind in KINDS
                            if kind in episode.streams.get(finger, {})
                            and event.get(kind + "_offset_bytes") is not None
                            and event.get(kind + "_length_bytes") is not None
                            and event.get(kind + "_shape") is not None
                        ],
                    }
                )
            sync_frames.append(sync_frame)

        payload = {
            "rollout_id": rollout_id,
            "camera": camera,
            "frame_min": frames[0],
            "frame_max": frames[-1],
            "sync_frames": sync_frames,
            "fingers": finger_series,
        }
        self._series_cache[cache_key] = payload
        return payload

    def _read_image(
        self,
        episode: EpisodeTactile,
        finger: str,
        event_id: Any,
        kind: str,
    ) -> tuple[bytes, list[int]] | None:
        event = self._event(episode, finger, event_id)
        stream = episode.streams.get(finger, {}).get(kind)
        if event is None or stream is None:
            return None
        offset_value = event.get(kind + "_offset_bytes")
        length_value = event.get(kind + "_length_bytes")
        shape_value = event.get(kind + "_shape")
        if offset_value is None or length_value is None or shape_value is None:
            return None
        offset = int(offset_value)
        length = int(length_value)
        shape = [int(value) for value in shape_value]
        if len(shape) != 2 or offset < 0 or length <= 0:
            return None
        height, width = shape
        if height <= 0 or width <= 0 or length != height * width:
            return None
        with stream.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise ValueError("short tactile image read")
        return data, shape

    def sprite(
        self,
        rollout_id: str,
        camera: str,
        video_frame: int,
        kind: str,
    ) -> bytes:
        if kind not in KINDS:
            raise KeyError("unknown tactile image kind")
        episode = self._episode(rollout_id)
        frames = episode.camera_frames.get(camera)
        rows = episode.camera_rows.get(camera)
        if not frames or not rows:
            raise KeyError("camera has no tactile synchronization")

        index = self._nearest_index(frames, int(video_frame))
        matched_frame = int(frames[index])
        cache_key = (rollout_id, camera, matched_frame, kind)
        with self._sprite_cache_lock:
            cached = self._sprite_cache.get(cache_key)
            if cached is not None:
                self._sprite_cache.move_to_end(cache_key)
                return cached

        row = episode.frames[rows[index]]
        tactile = row.get("tactile") if isinstance(row.get("tactile"), dict) else {}
        images: list[tuple[bytes, list[int]] | None] = []
        max_height = 0
        max_width = 0
        for finger in FINGERS:
            sync = tactile.get(finger) if isinstance(tactile.get(finger), dict) else {}
            image = self._read_image(episode, finger, sync.get("event_id"), kind)
            images.append(image)
            if image is not None:
                _data, shape = image
                max_height = max(max_height, shape[0])
                max_width = max(max_width, shape[1])

        if max_height <= 0 or max_width <= 0:
            raise KeyError("tactile sprite not found")

        total_width = max_width * len(FINGERS)
        sprite = bytearray(max_height * total_width)
        for finger_index, image in enumerate(images):
            if image is None:
                continue
            data, shape = image
            height, width = shape
            x_offset = finger_index * max_width
            for y in range(height):
                source_start = y * width
                target_start = y * total_width + x_offset
                sprite[target_start:target_start + width] = data[
                    source_start:source_start + width
                ]

        body = self._png(
            bytes(sprite),
            [max_height, total_width],
            compression_level=1,
        )
        with self._sprite_cache_lock:
            self._sprite_cache[cache_key] = body
            self._sprite_cache.move_to_end(cache_key)
            while len(self._sprite_cache) > self._sprite_cache_limit:
                self._sprite_cache.popitem(last=False)
        return body

    @staticmethod
    def _png_chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    @classmethod
    def _png(
        cls,
        data: bytes,
        shape: Any,
        compression_level: int = 3,
    ) -> bytes:
        if not isinstance(shape, (list, tuple)):
            raise ValueError("invalid tactile image shape")
        dims = [int(value) for value in shape]
        if len(dims) == 2:
            height, width = dims
            channels = 1
        elif len(dims) == 3 and dims[2] in (1, 3, 4):
            height, width, channels = dims
        else:
            raise ValueError("unsupported tactile image shape")
        if height <= 0 or width <= 0:
            raise ValueError("invalid tactile image dimensions")
        expected = height * width * channels
        if len(data) != expected:
            raise ValueError("tactile image byte length does not match shape")

        color_type = {1: 0, 3: 2, 4: 6}[channels]
        stride = width * channels
        scanlines = b"".join(
            b"\x00" + data[offset:offset + stride]
            for offset in range(0, len(data), stride)
        )
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        return (
            signature
            + cls._png_chunk(b"IHDR", ihdr)
            + cls._png_chunk(
                b"IDAT",
                zlib.compress(scanlines, level=compression_level),
            )
            + cls._png_chunk(b"IEND", b"")
        )

    def image(self, rollout_id: str, finger: str, event_id: str, kind: str) -> bytes:
        if finger not in FINGERS or kind not in KINDS:
            raise KeyError("unknown tactile image")
        episode = self._episode(rollout_id)
        event = self._event(episode, finger, event_id)
        stream = episode.streams.get(finger, {}).get(kind)
        if event is None or stream is None:
            raise KeyError("tactile image not found")

        offset = int(event[kind + "_offset_bytes"])
        length = int(event[kind + "_length_bytes"])
        shape = event[kind + "_shape"]
        if offset < 0 or length <= 0:
            raise ValueError("invalid tactile image byte range")
        with stream.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
        if len(data) != length:
            raise ValueError("short tactile image read")
        return self._png(data, shape)
