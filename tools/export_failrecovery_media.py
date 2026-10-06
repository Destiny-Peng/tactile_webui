#!/usr/bin/env python3
"""Export the selected RealRobot fail-recovery media for LF3R.

The source episode SQLite files and recordings are read only. Each exported
episode contains two RGB videos, five tactile byte streams, a tactile event
index, and a compact synchronized-frame index. Exported asset paths are relative
to PROJECT_ROOT, as expected by the LF3R annotator. Source recordings may live
outside PROJECT_ROOT; their provenance paths are then stored as absolute paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import ExitStack
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/mnt/hdd/qiuxia/datasets/failrecovery")
DATASET_ROOT = PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1"
EPISODE_ROOT = DATASET_ROOT / "failrecovery"
MANIFEST_PATH = DATASET_ROOT / "failrecovery_manifest.jsonl"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CAMERAS = {"cam_high": "realsense_color", "cam_wrist": "wrist_right"}
KNOWN_TASKS = {
    "Pick up the tube": (0, "tube"),
    "Pick up the usb stick and insert into the middle panel": (1, "usb_panel"),
    "Pick up the usb and insert it into the socket": (2, "usb_socket"),
}


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def source_path(path: Path) -> str:
    path = path.resolve()
    try:
        return rel(path)
    except ValueError:
        return str(path)


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height,codec_name,duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream: {path}")
    stream = streams[0]
    frames = stream.get("nb_frames")
    if frames in (None, "N/A"):
        command[5:5] = ["-count_frames"]
        command[command.index("stream=nb_frames,r_frame_rate,width,height,codec_name,duration")] = (
            "stream=nb_read_frames,nb_frames,r_frame_rate,width,height,codec_name,duration"
        )
        stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
        frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    return {
        "frames": int(frames),
        "fps": numerator / denominator,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream.get("codec_name"),
        "duration_seconds": float(stream.get("duration") or 0),
    }


def ensure_h264(path: Path, expected_frames: int | None = None) -> dict[str, Any]:
    before = probe_video(path)
    if expected_frames is not None and before["frames"] != expected_frames:
        raise ValueError(f"Video frame count changed: {path}")
    if before["codec"] == "h264":
        return before
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-map", "0:v:0", "-an",
                "-fps_mode", "passthrough", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(temporary),
            ],
            check=True,
        )
        after = probe_video(temporary)
        if after["codec"] != "h264" or after["frames"] != before["frames"]:
            raise ValueError(f"H.264 frame count mismatch: {path}")
        os.replace(temporary, path)
        return after
    finally:
        temporary.unlink(missing_ok=True)


def source_db(path: Path) -> sqlite3.Connection:
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"Source database has uncheckpointed WAL data: {path}")
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def export_tactile(connection: sqlite3.Connection, destination: Path) -> dict[str, Any]:
    tactile_dir = destination / "tactile"
    tactile_dir.mkdir()
    counts = {finger: 0 for finger in FINGERS}
    event_ids: set[int] = set()
    with ExitStack() as stack:
        index = stack.enter_context((tactile_dir / "events.jsonl").open("w", encoding="utf-8"))
        outputs = {
            finger: (
                stack.enter_context((tactile_dir / f"{finger}.raw.u8").open("wb")),
                stack.enter_context((tactile_dir / f"{finger}.deform.u8").open("wb")),
            )
            for finger in FINGERS
        }
        query = """
            SELECT e.id, e.source, e.sensor_ts_ns, e.receive_wall_ns,
                   e.receive_mono_ns, e.valid, e.source_time_status,
                   t.side, t.finger, t.channel, t.local_channel, t.f6_json,
                   t.raw_shape_json, t.deform_shape_json, t.raw_blob, t.deform_blob
            FROM events AS e JOIN tactile_frames AS t ON t.event_id = e.id
            WHERE e.source LIKE 'tactile:right:%'
            ORDER BY e.id
        """
        for row in connection.execute(query):
            finger = row["finger"]
            if finger not in outputs or row["side"] != "right":
                raise ValueError(f"Unexpected tactile source: {row['source']}")
            raw = row["raw_blob"]
            deform = row["deform_blob"]
            raw_shape = json.loads(row["raw_shape_json"])
            deform_shape = json.loads(row["deform_shape_json"])
            if raw is None or deform is None:
                raise ValueError(f"Missing tactile image at event {row['id']}")
            if len(raw) != raw_shape[0] * raw_shape[1]:
                raise ValueError(f"Unexpected tactile raw size at event {row['id']}")
            if len(deform) != deform_shape[0] * deform_shape[1]:
                raise ValueError(f"Unexpected tactile deform size at event {row['id']}")
            raw_file, deform_file = outputs[finger]
            record = {
                "event_id": row["id"], "finger": finger,
                "sample_index": counts[finger], "sensor_ts_ns": row["sensor_ts_ns"],
                "receive_wall_ns": row["receive_wall_ns"],
                "receive_mono_ns": row["receive_mono_ns"],
                "valid": bool(row["valid"]),
                "source_time_status": row["source_time_status"],
                "channel": row["channel"], "local_channel": row["local_channel"],
                "f6": json.loads(row["f6_json"]),
                "raw_shape": raw_shape, "deform_shape": deform_shape,
                "raw_offset_bytes": raw_file.tell(), "raw_length_bytes": len(raw),
                "deform_offset_bytes": deform_file.tell(),
                "deform_length_bytes": len(deform),
            }
            raw_file.write(raw)
            deform_file.write(deform)
            index.write(json_line(record))
            event_ids.add(row["id"])
            counts[finger] += 1
    return {"counts": counts, "event_ids": event_ids}


def export_sync_frames(
    connection: sqlite3.Connection, destination: Path, event_ids: set[int]
) -> dict[str, int]:
    frames = 0
    complete = 0
    missing_tactile = 0
    path = destination / "frames.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in connection.execute(
            "SELECT frame_index, elapsed_s, tick_wall_ns, tick_mono_ns, "
            "complete, stale_sources_json, snapshot_json "
            "FROM synchronized_frames ORDER BY frame_index"
        ):
            snapshot = json.loads(row["snapshot_json"])
            cameras = {}
            for key, source in CAMERAS.items():
                camera = snapshot.get(f"camera:{source}")
                cameras[key] = camera.get("frame_index") if camera else None
            tactile = {}
            for finger in FINGERS:
                item = snapshot.get(f"tactile:right:{finger}")
                event_id = item.get("event_id") if item else None
                if event_id is not None and event_id not in event_ids:
                    missing_tactile += 1
                tactile[finger] = {
                    "event_id": event_id,
                    "age_ms": item.get("age_ms") if item else None,
                    "stale": item.get("stale") if item else None,
                    "valid": item.get("valid") if item else None,
                }
            handle.write(json_line({
                "frame_index": row["frame_index"], "elapsed_s": row["elapsed_s"],
                "tick_wall_ns": row["tick_wall_ns"],
                "tick_mono_ns": row["tick_mono_ns"],
                "complete": bool(row["complete"]),
                "stale_sources": json.loads(row["stale_sources_json"]),
                "camera_frame_indices": cameras,
                "tactile": tactile,
            }))
            frames += 1
            complete += bool(row["complete"])
    if missing_tactile:
        raise ValueError(f"{missing_tactile} synchronized tactile references are missing")
    return {"frames": frames, "complete": complete}


def export_episode(
    source: Path, destination: Path, *, verify_videos: bool = True,
) -> dict[str, Any]:
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") not in (2, 3):
        raise ValueError(f"Unsupported recorder schema: {source}")
    if source_manifest.get("status") != "complete":
        raise ValueError(f"Incomplete source recording: {source}")
    signature = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "database_bytes": (source / "episode.sqlite3").stat().st_size,
        "video_bytes": {
            camera: (source / "videos" / f"{camera}.mp4").stat().st_size
            for camera in CAMERAS.values()
        },
    }
    if destination.exists():
        metadata_path = destination / "export.json"
        if not metadata_path.is_file():
            raise ValueError(f"Existing episode has no export metadata: {destination}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_signature") not in (None, signature):
            raise ValueError(f"Existing export belongs to different source data: {destination}")
        if metadata["synchronized"]["frames"] != source_manifest["synchronized_frame_count"]:
            raise ValueError(f"Existing export frame count differs from source: {destination}")
        for finger, count in metadata["tactile_counts"].items():
            if count != source_manifest["source_counts"][f"tactile:right:{finger}"]:
                raise ValueError(f"Existing tactile count differs from source: {destination}")
        metadata["source_manifest_path"] = source_path(source / "manifest.json")
        metadata["source_database_path"] = source_path(source / "episode.sqlite3")
        metadata["source_signature"] = signature
        metadata["recorder_schema_version"] = source_manifest["schema_version"]
        for key, camera in (CAMERAS.items() if verify_videos else ()):
            source_info = probe_video(source / "videos" / f"{camera}.mp4")
            for field in ("frames", "width", "height", "fps"):
                if source_info[field] != metadata["videos"][key][field]:
                    raise ValueError(f"Existing video differs from source: {destination} {camera}")
            metadata["videos"][key] = ensure_h264(
                destination / "videos" / f"{camera}.mp4",
                metadata["videos"][key]["frames"],
            )
        atomic_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        return metadata

    if not verify_videos:
        raise FileNotFoundError(f"Manifest refresh requires an existing export: {destination}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{source.name}.", dir=destination.parent))
    try:
        videos = temp_dir / "videos"
        videos.mkdir()
        video_info = {}
        for key, camera in CAMERAS.items():
            src = source / "videos" / f"{camera}.mp4"
            target = videos / src.name
            shutil.copy2(src, target)
            video_info[key] = ensure_h264(target)
        with source_db(source / "episode.sqlite3") as connection:
            tactile = export_tactile(connection, temp_dir)
            sync = export_sync_frames(connection, temp_dir, tactile["event_ids"])
            for key, camera in CAMERAS.items():
                count, maximum = connection.execute(
                    "SELECT COUNT(*), MAX(frame_index) FROM camera_frames WHERE source=?",
                    (f"camera:{camera}",),
                ).fetchone()
                if count != video_info[key]["frames"] or maximum != count - 1:
                    raise ValueError(f"Camera index/video count mismatch: {source.name} {camera}")
        if sync["frames"] != source_manifest["synchronized_frame_count"]:
            raise ValueError(f"Synchronized frame count mismatch: {source}")
        metadata = {
            "schema_version": 1, "episode": source.name,
            "source_manifest_path": source_path(source / "manifest.json"),
            "source_database_path": source_path(source / "episode.sqlite3"),
            "source_signature": signature,
            "recorder_schema_version": source_manifest["schema_version"],
            "videos": video_info, "synchronized": sync,
            "tactile_counts": tactile["counts"],
            "tactile_encoding": "uint8 row-major; byte offsets and shapes in tactile/events.jsonl",
            "tactile_fingers": list(FINGERS),
        }
        (temp_dir / "export.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, destination)
        return metadata
    except BaseException:
        shutil.rmtree(temp_dir)
        raise


def filename_outcome(name: str) -> str | None:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    success = bool(tokens & {"success", "successful"})
    failure = bool(tokens & {"fail", "failure", "failed", "mixedfail"})
    if success == failure:
        return None
    return "success" if success else "failure"


def manifest_record(
    source: Path, destination: Path, index: int, meta: dict[str, Any],
    task_id: int, task_key: str, previous_id: str | None = None,
) -> dict[str, Any]:
    recorded = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    primary = meta["videos"]["cam_high"]
    identity = {
        key: recorded.get(key) for key in (
            "episode_label", "recording_started_at_utc", "recording_start_wall_ns",
            "recorder_git_commit", "task",
        )
    }
    digest = hashlib.sha1(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    review_path = source / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    note = str(review.get("notes") or "").strip().lower()
    review_outcome = {"success": "success", "failure": "failure", "fail": "failure"}.get(note)
    filename_hint = filename_outcome(source.name)
    outcome = review_outcome or filename_hint or "unknown"
    return {
        "schema_version": 1,
        "id": previous_id or f"realrobot-failrecovery-{source.name}-{digest}",
        "task_suite": "realrobot_failrecovery", "task_id": task_id,
        "task_key": task_key,
        "episode_index": index, "episode_label": recorded["episode_label"],
        "task": recorded["task"], "task_description": recorded["task"],
        "ground_truth_outcome": outcome,
        "outcome_source": (
            "review.json" if review_outcome else "filename" if filename_hint else None
        ),
        "review": review,
        "source_review_path": source_path(review_path) if review_path.exists() else None,
        "episode_label_outcome_hint": filename_hint,
        "quality_flags": (
            ["short_recording"] if recorded["synchronized_frame_count"] < 2 * recorded["config"]["sample_hz"] else []
        ),
        "source_kind": "realrobot_failrecovery",
        "analysis_partition": "natural_observation",
        "dataset_role": "realrobot_failrecovery",
        "camera_video_paths": {
            key: rel(destination / "videos" / f"{camera}.mp4")
            for key, camera in CAMERAS.items()
        },
        "total_frames": primary["frames"], "fps": primary["fps"],
        "duration_seconds": primary["duration_seconds"],
        "video_width": primary["width"], "video_height": primary["height"],
        "video_codec": primary["codec"],
        "synchronized_frame_count": meta["synchronized"]["frames"],
        "complete_synchronized_frame_count": meta["synchronized"]["complete"],
        "synchronized_frames_path": rel(destination / "frames.jsonl"),
        "tactile_events_path": rel(destination / "tactile/events.jsonl"),
        "tactile_stream_paths": {
            finger: {
                "raw": rel(destination / "tactile" / f"{finger}.raw.u8"),
                "deform": rel(destination / "tactile" / f"{finger}.deform.u8"),
            }
            for finger in FINGERS
        },
        "tactile_counts": meta["tactile_counts"],
        "sample_hz": recorded["config"]["sample_hz"],
        "recorded_duration_seconds": recorded["duration_s"],
        "recording_status": recorded["status"],
        "source_manifest_path": source_path(source / "manifest.json"),
        "recorder_schema_version": recorded["schema_version"],
        "recording_started_at_utc": recorded.get("recording_started_at_utc"),
        "recorder_git_commit": recorded.get("recorder_git_commit"),
        "policy_family": "realrobot_recording", "policy_checkpoint": None,
        "csv_path": None, "first_environment_timestep": None,
        "last_environment_timestep": None,
    }


def save_goal_image(record: dict[str, Any], path: Path) -> dict[str, Any]:
    video = PROJECT_ROOT / record["camera_video_paths"]["cam_high"]
    final_frame = int(record["total_frames"]) - 1
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video), "-vf", f"select=eq(n\\,{final_frame})",
                "-vsync", "0", "-frames:v", "1", str(temporary),
            ],
            check=True,
        )
        if temporary.stat().st_size == 0:
            raise ValueError(f"No final frame extracted from {video}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": rel(path), "source_rollout_id": record["id"],
        "camera": "cam_high", "video_frame_index": final_frame,
        "task_id": record["task_id"], "task_key": record["task_key"],
        "ground_truth_outcome": record["ground_truth_outcome"],
        "outcome_source": record["outcome_source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=SOURCE_ROOT,
        help="Directory containing episode folders; may be outside PROJECT_ROOT.",
    )
    parser.add_argument("--output-root", type=Path, default=EPISODE_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--manifests-only", action="store_true",
        help="Refresh manifests from existing exports without copying or transcoding videos.",
    )
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (output_root, manifest_path):
        path.relative_to(PROJECT_ROOT)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    all_sources = sorted(path for path in source_root.iterdir() if path.is_dir())
    sources = []
    excluded = []
    tasks = set()
    for source in all_sources:
        manifest = source / "manifest.json"
        if not manifest.is_file():
            excluded.append({"episode": source.name, "reason": "missing_manifest"})
            continue
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        review_path = source / "review.json"
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
        if str(review.get("notes") or "").strip().lower().startswith("discard"):
            excluded.append({"episode": source.name, "reason": "review_discard", "review": review})
            continue
        if recorded.get("status") != "complete" or recorded.get("worker_errors"):
            excluded.append({"episode": source.name, "reason": "incomplete_or_worker_errors"})
            continue
        if recorded.get("schema_version") not in (2, 3):
            raise ValueError(f"Unsupported recorder schema: {source}")
        for required in [source / "episode.sqlite3", *(source / "videos" / f"{c}.mp4" for c in CAMERAS.values())]:
            if not required.is_file():
                raise FileNotFoundError(required)
        sources.append(source)
        tasks.add(recorded["task"])
    if not sources:
        raise ValueError(f"No episodes under {source_root}")
    task_catalog = {}
    next_task_id = max(v[0] for v in KNOWN_TASKS.values()) + 1
    for task in sorted(tasks):
        if task in KNOWN_TASKS:
            task_catalog[task] = KNOWN_TASKS[task]
        else:
            task_catalog[task] = (next_task_id, f"task_{next_task_id}")
            next_task_id += 1
    previous_ids = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            videos = old.get("camera_video_paths") or {}
            if videos:
                episode = Path(next(iter(videos.values()))).parents[1].name
                previous_ids[episode] = old["id"]
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(sources):
        destination = output_root / source.name
        print(f"[{index + 1}/{len(sources)}] {source.name}", flush=True)
        metadata = export_episode(source, destination, verify_videos=not args.manifests_only)
        task = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["task"]
        task_id, task_key = task_catalog[task]
        records.append(manifest_record(
            source, destination, index, metadata, task_id, task_key,
            previous_ids.get(source.name),
        ))
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate rollout IDs")
    goals = {}
    missing_goal_tasks = []
    goal_root = output_root / "goal_images"
    goal_root.mkdir(parents=True, exist_ok=True)
    for task, (task_id, task_key) in task_catalog.items():
        task_records = [r for r in records if r["task_id"] == task_id]
        successes = [r for r in task_records if r["ground_truth_outcome"] == "success"]
        reviewed = [r for r in successes if r["outcome_source"] == "review.json"]
        candidates = reviewed or successes
        selection_reason = "success_rollout"
        if not candidates and task_key == "tube":
            # The dataset owner permits any tube rollout as the goal source.
            candidates = [r for r in task_records if r["ground_truth_outcome"] == "unknown"]
            candidates = [r for r in candidates if not r["quality_flags"]] or candidates
            selection_reason = "user_allowed_unknown_tube_rollout"
        if not candidates:
            missing_goal_tasks.append(task_key)
            continue
        goal_record = candidates[0]
        goals[task_key] = save_goal_image(goal_record, goal_root / f"{task_key}.png")
        goals[task_key]["selection_reason"] = selection_reason
    for record in records:
        if record["task_key"] in goals:
            record["goal_image_path"] = goals[record["task_key"]]["path"]
    atomic_text(manifest_path, "".join(json_line(record) for record in records))
    task_manifests = {}
    task_manifest_root = output_root / "task_manifests"
    archive_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for task, (task_id, task_key) in task_catalog.items():
        subset = [r for r in records if r["task_id"] == task_id]
        task_filename = f"{manifest_path.stem}.{task_key}.jsonl"
        task_manifest = task_manifest_root / task_filename
        legacy_manifest = manifest_path.with_name(task_filename)
        if legacy_manifest.exists():
            archive = task_manifest_root / "previous" / archive_stamp / task_filename
            archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(legacy_manifest, archive)
        atomic_text(task_manifest, "".join(json_line(r) for r in subset))
        task_manifests[task_key] = {
            "task_id": task_id, "task_description": task,
            "manifest": rel(task_manifest), "rollouts": len(subset),
            "outcome_counts": dict(Counter(r["ground_truth_outcome"] for r in subset)),
        }
    summary = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_root": source_path(source_root), "episode_root": rel(output_root),
        "manifest": rel(manifest_path), "total_rollouts": len(records),
        "source_episode_count": len(all_sources),
        "excluded_episodes": excluded,
        "task_manifests": task_manifests,
        "manifest_layout": "one top-level combined manifest; task subsets under episode_root/task_manifests",
        "recorder_schema_counts": dict(Counter(r["recorder_schema_version"] for r in records)),
        "outcome_counts": dict(Counter(r["ground_truth_outcome"] for r in records)),
        "outcome_source_counts": dict(Counter(r["outcome_source"] or "unknown" for r in records)),
        "short_episodes": [r["id"] for r in records if "short_recording" in r["quality_flags"]],
        "camera_keys": list(CAMERAS), "tactile_fingers": list(FINGERS),
        "synchronized_frames": sum(r["synchronized_frame_count"] for r in records),
        "complete_synchronized_frames": sum(
            r["complete_synchronized_frame_count"] for r in records
        ),
        "tactile_samples": sum(
            sum(r["tactile_counts"].values()) for r in records
        ),
        "goal_images": goals,
        "missing_goal_tasks": missing_goal_tasks,
    }
    atomic_text(
        manifest_path.with_name(manifest_path.stem + ".summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_text(output_root / "README.md", (
        "# RealRobot failrecovery export\n\n"
        f"Aggregate manifest: `{rel(manifest_path)}` (paths relative to LF3R).\n"
        f"Episodes: {len(records)}; excluded: {len(excluded)}.\n\n"
        "Only the combined manifest lives in the top-level discovery directory. "
        "Task subsets live under task_manifests/ so the WebUI will not reject "
        "duplicate IDs across combined and subset manifests.\n\n"
        "Each episode has H.264 realsense_color (cam_high) and wrist_right (cam_wrist), "
        "frames.jsonl, tactile/events.jsonl, and raw/deform byte streams for five fingers. "
        "Image bytes are row-major uint8. Use the event index byte offsets, lengths, and "
        "shapes to read images. Join synchronized frames to events by event_id; video "
        "frame indices may differ from synchronized frame indices.\n\n"
        "Review notes success/failure/fail take priority for ground_truth_outcome. "
        "Otherwise filename tokens success/successful imply success, and "
        "fail/failure/failed/mixedfail imply failure. The mixedfail token is a "
        "filename-based inference, not an individually reviewed result. Names with "
        "no outcome token or conflicting tokens remain unknown. outcome_source "
        "records review.json or filename; filename hints are stored separately. Discard reviews "
        "are excluded. Short recordings are retained with quality_flags.\n\n"
        "Each task with a success rollout has goal_images/<task_key>.png, extracted "
        "from its last cam_high frame. Reviewed successes take priority over filename "
        "labels. For tube only, the dataset owner permits an unknown-outcome rollout "
        "when no success is available; its outcome remains unknown. Other tasks "
        "without a success rollout have no goal image. Matching-task "
        "records declare goal_image_path. Source details and per-task manifests are listed "
        "in the aggregate summary JSON.\n\n"
        "Regenerate with:\n\n```bash\n"
        f"python3 tools/export_failrecovery_media.py --source-root {str(source_root)!r}\n"
        "```\n\nRefresh manifests using existing exports with `--manifests-only`.\n"
    ))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
