"""FailRecovery tactile reader for the standalone tactile WebUI."""

from __future__ import annotations

import bisect
import json
import sqlite3
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
    sqlite_path: Path | None = None


class FailRecoveryTactileService:
    """Small, dataset-specific index for failrecovery_manifest.jsonl."""

    def __init__(
        self,
        project_root: Path,
        manifest_path: Path | None = None,
        *,
        episode_cache_limit: int = 2,
        series_cache_limit: int = 2,
        manifest_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.manifest_path = (manifest_path or (
            self.project_root
            / "datasets"
            / "failrecovery"
            / "manifest.jsonl"
        )).expanduser().resolve()
        # The manifest may itself be reached through a symlink from this
        # standalone repository and therefore live outside project_root.
        # Only payload paths read from the manifest are constrained to
        # project_root by _project_file().
        # Episode and expanded-series objects are the largest long-lived
        # allocations in this viewer. Keep only the most recently used pair of
        # rollouts so memory plateaus while browsing a long dataset.
        self.episodes: OrderedDict[str, EpisodeTactile] = OrderedDict()
        self._series_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._episode_cache_limit = max(1, int(episode_cache_limit))
        self._series_cache_limit = max(1, int(series_cache_limit))
        self._episode_cache_hits = 0
        self._episode_cache_misses = 0
        self._episode_cache_evictions = 0
        self._series_cache_hits = 0
        self._series_cache_misses = 0
        self._series_cache_evictions = 0
        self._series_locks: dict[tuple[str, str], threading.Lock] = {}
        self._sprite_cache: OrderedDict[tuple[str, str, int, str], bytes] = OrderedDict()
        self._sprite_cache_lock = threading.Lock()
        self._sprite_cache_limit = 96
        self._manifest_rows: dict[str, dict[str, Any]] = {}
        self._episode_locks: dict[str, threading.Lock] = {}
        self._image_cache: OrderedDict[tuple[str, str, str, str], bytes] = OrderedDict()
        self._image_cache_limit = 128
        self._image_cache_lock = threading.Lock()
        if manifest_rows is None:
            self._load_manifest()
        else:
            self._register_manifest_rows(manifest_rows)

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
        self._register_manifest_rows(manifest_rows)

    def _register_manifest_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            rollout_id = str(row.get("id") or "").strip()
            if rollout_id:
                self._manifest_rows[rollout_id] = row
                self._episode_locks[rollout_id] = threading.Lock()

    def _sqlite_database(self, row: dict[str, Any]) -> Path | None:
        value = row.get("sqlite_path") or row.get("source_database_path")
        if not isinstance(value, str) or not value.strip():
            return None
        raw = Path(value).expanduser()
        path = raw.resolve() if raw.is_absolute() else self._project_file(value)
        # The source manifest is a trusted local file, not a URL parameter.
        # Historical LF3R manifests may point to the recorder DB outside the
        # exported LF3R repository; keep that original location read-only.
        return path if path.is_file() else None

    @staticmethod
    def _sqlite_connect(path: Path) -> sqlite3.Connection:
        # mode=ro does not create the database or make modifications. Unlike
        # immutable=1, it is safe when the recorder has not checkpointed WAL.
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _load_sqlite_episode(self, rollout_id: str, database: Path) -> EpisodeTactile:
        # Avoid SELECT *: never load large raw/deform BLOBs while building
        # episode synchronization and F6 history.
        with self._sqlite_connect(database) as connection:
            frames: list[dict[str, Any]] = []
            for frame in connection.execute(
                "SELECT frame_index, elapsed_s, tick_wall_ns, tick_mono_ns, "
                "complete, snapshot_json FROM synchronized_frames ORDER BY frame_index"
            ):
                snapshot = json.loads(frame["snapshot_json"])
                cameras = {}
                for camera, source in (
                    ("cam_high", "realsense_color"), ("cam_wrist", "wrist_right")
                ):
                    item = snapshot.get("camera:" + source)
                    cameras[camera] = item.get("frame_index") if isinstance(item, dict) else None
                tactile = {}
                for finger in FINGERS:
                    item = snapshot.get("tactile:right:" + finger)
                    item = item if isinstance(item, dict) else {}
                    tactile[finger] = {
                        "event_id": item.get("event_id"),
                        "age_ms": item.get("age_ms"),
                        "stale": item.get("stale"),
                        "valid": item.get("valid"),
                    }
                frames.append({
                    "frame_index": frame["frame_index"],
                    "elapsed_s": frame["elapsed_s"],
                    "tick_wall_ns": frame["tick_wall_ns"],
                    "tick_mono_ns": frame["tick_mono_ns"],
                    "complete": bool(frame["complete"]),
                    "camera_frame_indices": cameras,
                    "tactile": tactile,
                })

            events: dict[str, dict[str, Any]] = {}
            finger_events: dict[tuple[str, str], dict[str, Any]] = {}
            query = (
                "SELECT e.id, e.sensor_ts_ns, e.receive_wall_ns, e.receive_mono_ns, "
                "e.valid, t.rowid AS tactile_rowid, t.finger, t.f6_json, "
                "t.raw_shape_json, t.deform_shape_json, "
                "length(t.raw_blob) AS raw_bytes, length(t.deform_blob) AS deform_bytes "
                "FROM tactile_frames t JOIN events e ON e.id=t.event_id "
                "WHERE e.source LIKE 'tactile:right:%' AND t.side='right' ORDER BY e.id"
            )
            for item in connection.execute(query):
                finger = str(item["finger"]).lower()
                if finger not in FINGERS:
                    continue
                event_id = str(item["id"])
                value: dict[str, Any] = {
                    "event_id": item["id"], "finger": finger,
                    "sensor_ts_ns": item["sensor_ts_ns"],
                    "receive_wall_ns": item["receive_wall_ns"],
                    "receive_mono_ns": item["receive_mono_ns"],
                    "valid": bool(item["valid"]),
                    "f6": json.loads(item["f6_json"]) if item["f6_json"] else None,
                    "_sqlite_rowid": item["tactile_rowid"],
                }
                for kind in KINDS:
                    raw_shape = item[kind + "_shape_json"]
                    length = item[kind + "_bytes"]
                    if raw_shape and length is not None:
                        value[kind + "_shape"] = json.loads(raw_shape)
                        value[kind + "_length_bytes"] = length
                        value[kind + "_offset_bytes"] = 0
                events[event_id] = value
                finger_events[(finger, event_id)] = value

        camera_pairs: dict[str, list[tuple[int, int]]] = {}
        for index, frame in enumerate(frames):
            for camera, value in frame["camera_frame_indices"].items():
                if value is not None:
                    camera_pairs.setdefault(camera, []).append((int(value), index))
        camera_frames: dict[str, list[int]] = {}
        camera_rows: dict[str, list[int]] = {}
        for camera, pairs in camera_pairs.items():
            pairs.sort()
            camera_frames[camera] = [x for x, _ in pairs]
            camera_rows[camera] = [index for _, index in pairs]
        # A stand-in for existing image_kinds checking in frame() and series().
        streams = {finger: {kind: database for kind in KINDS} for finger in FINGERS}
        return EpisodeTactile(
            rollout_id=rollout_id, frames=frames, camera_frames=camera_frames,
            camera_rows=camera_rows, events=events, finger_events=finger_events,
            streams=streams, sqlite_path=database,
        )

    def _load_episode(self, rollout_id: str, row: dict[str, Any]) -> EpisodeTactile:
        database = self._sqlite_database(row)
        if database is not None:
            return self._load_sqlite_episode(rollout_id, database)
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

        with self._cache_lock:
            episode = self.episodes.get(rollout_id)
            if episode is not None:
                self.episodes.move_to_end(rollout_id)
                self._episode_cache_hits += 1
                return episode

        # Separate locks let unrelated episodes load concurrently. Recheck
        # after acquiring the per-episode lock so concurrent first requests
        # still perform only one JSONL load.
        with self._episode_locks[rollout_id]:
            with self._cache_lock:
                episode = self.episodes.get(rollout_id)
                if episode is not None:
                    self.episodes.move_to_end(rollout_id)
                    self._episode_cache_hits += 1
                    return episode
            try:
                episode = self._load_episode(rollout_id, row)
            except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
                raise KeyError("tactile rollout not found") from exc
            with self._cache_lock:
                self._episode_cache_misses += 1
                self.episodes[rollout_id] = episode
                self.episodes.move_to_end(rollout_id)
                while len(self.episodes) > self._episode_cache_limit:
                    self.episodes.popitem(last=False)
                    self._episode_cache_evictions += 1
            return episode

    def cache_stats(self) -> dict[str, Any]:
        with self._cache_lock:
            episode_stats = {
                "size": len(self.episodes),
                "limit": self._episode_cache_limit,
                "hits": self._episode_cache_hits,
                "misses": self._episode_cache_misses,
                "evictions": self._episode_cache_evictions,
                "rollouts": list(self.episodes.keys()),
            }
            series_stats = {
                "size": len(self._series_cache),
                "limit": self._series_cache_limit,
                "hits": self._series_cache_hits,
                "misses": self._series_cache_misses,
                "evictions": self._series_cache_evictions,
                "keys": [
                    {"rollout_id": rollout_id, "camera": camera}
                    for rollout_id, camera in self._series_cache.keys()
                ],
            }
        with self._sprite_cache_lock:
            sprite_stats = {
                "size": len(self._sprite_cache),
                "limit": self._sprite_cache_limit,
            }
        return {
            "episodes": episode_stats,
            "series": series_stats,
            "sprites": sprite_stats,
        }

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
        with self._cache_lock:
            cached = self._series_cache.get(cache_key)
            if cached is not None:
                self._series_cache.move_to_end(cache_key)
                self._series_cache_hits += 1
                return cached
            series_lock = self._series_locks.setdefault(cache_key, threading.Lock())

        # Building a series duplicates a sizeable subset of the episode index.
        # Avoid duplicate builds if several UI requests arrive together.
        with series_lock:
            with self._cache_lock:
                cached = self._series_cache.get(cache_key)
                if cached is not None:
                    self._series_cache.move_to_end(cache_key)
                    self._series_cache_hits += 1
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
            with self._cache_lock:
                self._series_cache_misses += 1
                self._series_cache[cache_key] = payload
                self._series_cache.move_to_end(cache_key)
                while len(self._series_cache) > self._series_cache_limit:
                    self._series_cache.popitem(last=False)
                    self._series_cache_evictions += 1
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
        if episode.sqlite_path is not None:
            rowid = event.get("_sqlite_rowid")
            if rowid is None:
                return None
            with self._sqlite_connect(episode.sqlite_path) as connection:
                result = connection.execute(
                    f"SELECT {kind}_blob FROM tactile_frames WHERE rowid=?", (rowid,)
                ).fetchone()
            data = result[0] if result else b""
        else:
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

    def static_image_path(self, rollout_id: str, finger: str, event_id: str, kind: str) -> Path | None:
        if finger not in FINGERS or kind not in KINDS:
            raise KeyError("unknown tactile image")
        episode = self._episode(rollout_id)
        if episode.sqlite_path is not None:
            return None
        event = self._event(episode, finger, event_id)
        stream = episode.streams.get(finger, {}).get(kind)
        if event is None or stream is None:
            raise KeyError("tactile image not found")
        sample_index = event.get("sample_index")
        if sample_index is None:
            return None
        path = stream.parent / "images" / kind / finger / f"{int(sample_index):06d}.png"
        return path if path.is_file() else None

    def image(self, rollout_id: str, finger: str, event_id: str, kind: str) -> bytes:
        if finger not in FINGERS or kind not in KINDS:
            raise KeyError("unknown tactile image")
        cache_key = (rollout_id, finger, str(event_id), kind)
        with self._image_cache_lock:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache.move_to_end(cache_key)
                return cached
        episode = self._episode(rollout_id)
        if episode.sqlite_path is not None:
            result = self._read_image(episode, finger, event_id, kind)
            if result is None:
                raise KeyError("tactile image not found")
            pixels, shape = result
            body = self._png(pixels, shape, compression_level=3)
            with self._image_cache_lock:
                self._image_cache[cache_key] = body
                self._image_cache.move_to_end(cache_key)
                while len(self._image_cache) > self._image_cache_limit:
                    self._image_cache.popitem(last=False)
            return body
        event = self._event(episode, finger, event_id)
        stream = episode.streams.get(finger, {}).get(kind)
        if event is None or stream is None:
            raise KeyError("tactile image not found")

        static_path = self.static_image_path(rollout_id, finger, event_id, kind)
        if static_path is not None:
            return static_path.read_bytes()

        # Compatibility fallback for an old export that has not been
        # materialized yet. Canonical migrated/new exports use static PNGs.
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


class MultiTactileService:
    """Route existing tactile endpoints to the correct source-manifest reader.

    Each reader has its own original data root. All source files stay in place.
    """

    def __init__(self, sources: list[tuple[Path, Path, list[dict[str, Any]]]]) -> None:
        self._by_rollout: dict[str, FailRecoveryTactileService] = {}
        self._readers: list[FailRecoveryTactileService] = []
        for manifest, root, rows in sources:
            if not rows:
                continue
            reader = FailRecoveryTactileService(root, manifest, manifest_rows=rows)
            for row in rows:
                rollout_id = str(row["id"])
                if rollout_id in self._by_rollout:
                    raise ValueError(f"Duplicate rollout id across tactile readers: {rollout_id}")
                self._by_rollout[rollout_id] = reader
            self._readers.append(reader)

    def has_rollout(self, rollout_id: str) -> bool:
        return rollout_id in self._by_rollout

    def _reader(self, rollout_id: str) -> FailRecoveryTactileService:
        try:
            return self._by_rollout[rollout_id]
        except KeyError as exc:
            raise KeyError("tactile rollout not found") from exc

    def frame(self, rollout_id: str, camera: str, video_frame: int) -> dict[str, Any]:
        return self._reader(rollout_id).frame(rollout_id, camera, video_frame)

    def series(self, rollout_id: str, camera: str) -> dict[str, Any]:
        return self._reader(rollout_id).series(rollout_id, camera)

    def sprite(self, rollout_id: str, camera: str, video_frame: int, kind: str) -> bytes:
        return self._reader(rollout_id).sprite(rollout_id, camera, video_frame, kind)

    def image(self, rollout_id: str, finger: str, event_id: str, kind: str) -> bytes:
        return self._reader(rollout_id).image(rollout_id, finger, event_id, kind)

    def cache_stats(self) -> dict[str, Any]:
        return {
            "readers": len(self._readers),
            "sources": [reader.cache_stats() for reader in self._readers],
        }
