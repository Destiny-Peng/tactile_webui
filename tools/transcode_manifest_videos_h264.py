#!/usr/bin/env python3
"""Batch H.264 transcode selected WebUI manifest videos into a local cache.

Reads a trusted, server-generated plan JSON. Never modifies source videos.
State and log are persisted per run; cached videos are reused by the WebUI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "videos" / "h264"


def cache_path(source: Path, cache_dir: Path = CACHE_DIR) -> Path:
    """Stable across restarts; changes whenever the original file changes."""
    source = source.expanduser().resolve()
    info = source.stat()
    identity = f"{source}\0{info.st_size}\0{info.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.h264.mp4"


def probe(source: Path) -> dict:
    finished = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,nb_frames",
         "-of", "json", str(source)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(finished.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"no video stream in {source}")
    return streams[0]


def compatible_video(path: Path, cache_dir: Path = CACHE_DIR) -> Path:
    """Pick an already validated H.264 cache, otherwise original source."""
    try:
        destination = cache_path(path, cache_dir)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
    except OSError:
        pass
    return path


def atomic_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run(plan_path: Path, state_path: Path, cache_dir: Path = CACHE_DIR) -> int:
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    videos = [Path(value).expanduser().resolve() for value in plan["videos"]]
    if len(videos) != len(set(videos)):
        raise ValueError("duplicate videos in batch plan")
    # The plan is created by the server from its loaded manifest catalog.
    state = {
        "status": "running", "pid": os.getpid(), "total": len(videos),
        "processed": 0, "converted": 0, "already_h264": 0, "cached": 0,
        "failed": 0, "missing": 0, "current": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifests": plan.get("manifests", []), "errors": [],
    }
    atomic_json(state_path, state)
    print(f"BATCH_H264_SELECTED unique_videos={len(videos)}", flush=True)

    for index, video in enumerate(videos, 1):
        state["current"] = str(video)
        atomic_json(state_path, state)
        print(f"[{index}/{len(videos)}] {video}", flush=True)
        temporary = None
        try:
            if not video.is_file():
                state["missing"] += 1
                print(f"  MISSING {video}", flush=True)
                continue
            source_info = probe(video)
            if source_info.get("codec_name") == "h264" and source_info.get("pix_fmt") == "yuv420p":
                state["already_h264"] += 1
                print("  SKIP already H.264/yuv420p", flush=True)
                continue
            output = cache_path(video, cache_dir)
            if output.is_file() and output.stat().st_size:
                state["cached"] += 1
                print(f"  SKIP cache exists: {output}", flush=True)
                continue

            cache_dir.mkdir(parents=True, exist_ok=True)
            # Ensure output can coexist with the original; insufficient disk
            # space fails safely without replacing or moving the original.
            available = shutil.disk_usage(cache_dir).free
            reserve = max(512 * 1024 * 1024, video.stat().st_size * 2)
            if available < reserve:
                raise OSError(
                    f"insufficient free space in cache: {available} bytes; "
                    f"require at least {reserve} bytes"
                )
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.partial.mp4")
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video), "-map", "0:v:0", "-an",
                "-fps_mode", "passthrough", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(temporary),
            ]
            start = time.perf_counter()
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                raise RuntimeError(f"ffmpeg exited with code {completed.returncode}")
            converted = probe(temporary)
            if converted.get("codec_name") != "h264" or converted.get("pix_fmt") != "yuv420p":
                raise ValueError("unexpected output codec or pixel format")
            if (source_info.get("nb_frames") not in (None, "N/A")
                and converted.get("nb_frames") not in (None, "N/A")
                and int(source_info["nb_frames"]) != int(converted["nb_frames"])):
                raise ValueError("output frame count differs from original")
            # Source may have changed while encoding; never publish a stale file.
            if cache_path(video, cache_dir) != output:
                raise ValueError("source video changed during transcode")
            os.replace(temporary, output)
            temporary = None
            state["converted"] += 1
            print(f"  CONVERTED {output} in {time.perf_counter()-start:.1f}s", flush=True)
        except Exception as exc:
            state["failed"] += 1
            message = f"{video}: {exc}"
            state["errors"].append(message)
            print(f"  FAILED {message}", file=sys.stderr, flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            state["processed"] = index
            atomic_json(state_path, state)

    state["status"] = "failed" if state["failed"] or state["missing"] else "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, state)
    print("BATCH_H264_SUMMARY " + " ".join(
        f"{key}={state[key]}" for key in
        ("total", "converted", "already_h264", "cached", "missing", "failed")
    ), flush=True)
    return 1 if state["status"] == "failed" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()
    try:
        return run(args.plan, args.state, args.cache_dir)
    except Exception as exc:
        atomic_json(args.state, {"status": "failed", "error": str(exc)})
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
