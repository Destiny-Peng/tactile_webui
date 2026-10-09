#!/usr/bin/env python3
"""Index original recorder episode.sqlite3 + MP4 files without exporting payloads.

Usage:
  python tools/index_sqlite_dataset.py /path/to/failrecovery --name raw_failrecovery
Creates datasets/<name>/manifest.jsonl inside the standalone repo.
No original SQLite, image BLOB, frame, or video is copied or modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = {"cam_high": "realsense_color", "cam_wrist": "wrist_right"}


def video_probe(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,nb_frames",
        "-of", "json", str(path)
    ], check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise ValueError(f"no video stream: {path}")
    data = streams[0]
    rate = data.get("r_frame_rate", "0/1")
    a, b = (int(item) for item in rate.split("/"))
    return {
        "fps": a / b if b else 0,
        "width": data.get("width"),
        "height": data.get("height"),
        "codec": data.get("codec_name"),
        "frames": int(data["nb_frames"]) if str(data.get("nb_frames") or "").isdigit() else None,
    }


def build_one(episode: Path, source_root: Path, dataset: str) -> dict:
    metadata = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("recorder episode status is not complete")
    database = episode / "episode.sqlite3"
    videos = {key: episode / "videos" / f"{source}.mp4" for key, source in CAMERAS.items()}
    for required in (database, *videos.values()):
        if not required.is_file():
            raise FileNotFoundError(required)
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        count = db.execute("SELECT COUNT(*) FROM synchronized_frames").fetchone()[0]
        if count < 1:
            raise ValueError("no synchronized frames")
        camera_count = db.execute(
            "SELECT COUNT(*) FROM camera_frames WHERE source=?", ("camera:realsense_color",)
        ).fetchone()[0]
    camera = video_probe(videos["cam_high"])
    review_path = episode / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
    notes = str(review.get("notes") or "").strip().lower()
    outcome = "success" if notes == "success" else "failure" if notes in {"fail", "failure"} else "unknown"
    rel = episode.relative_to(source_root).as_posix()
    task = str(metadata.get("task") or "tactile_task")
    return {
        "schema_version": 1,
        "id": f"{dataset}--{episode.name}",
        "task_suite": dataset,
        "task_key": task,
        "task_description": task,
        "episode_label": metadata.get("episode_label", episode.name),
        "ground_truth_outcome": outcome,
        "dataset_role": "realrobot_failrecovery",
        "source_kind": "realrobot_failrecovery",
        "data_root": str(source_root.resolve()),
        "source_database_path": f"{rel}/episode.sqlite3",
        "camera_video_paths": {
            key: f"{rel}/videos/{src}.mp4" for key, src in CAMERAS.items()
        },
        "total_frames": camera["frames"] or camera_count,
        "fps": camera["fps"],
        "video_width": camera["width"],
        "video_height": camera["height"],
        "video_codec": camera["codec"],
        "synchronized_frame_count": count,
        "review": review,
        "recording_status": metadata.get("status"),
    }


def index_dataset(source_root: Path, output: Path, dataset: str, overwrite: bool = False) -> list[dict]:
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().absolute()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output.is_symlink():
        raise ValueError("Refusing to replace a symlinked manifest")
    if output.exists() and not overwrite:
        raise ValueError(f"Manifest already exists: {output} (use --overwrite)")
    # Only explicitly existing episode directories, never follow recursive links.
    episodes = sorted(p for p in source_root.iterdir() if p.is_dir() and (p / "episode.sqlite3").is_file())
    if not episodes:
        raise ValueError(f"No episode.sqlite3 files in {source_root}")
    rows = []
    skipped = []
    for episode in episodes:
        try:
            rows.append(build_one(episode, source_root, dataset))
        except (OSError, ValueError, sqlite3.Error, subprocess.CalledProcessError) as exc:
            skipped.append((episode.name, str(exc)))
            print(f"[skip] {episode.name}: {exc}", flush=True)
    if not rows:
        raise ValueError("All episodes failed indexing; refusing to create manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    summary = {
        "dataset": dataset, "source_root": str(source_root),
        "created_at": datetime.now().astimezone().isoformat(),
        "episodes_indexed": len(rows), "episodes_skipped": skipped,
        "manifest": str(output)
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Indexed {len(rows)} original episodes; skipped {len(skipped)}")
    print(f"Manifest: {output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing recorder episode directories")
    parser.add_argument("--name", required=True, help="Dataset name / manifest subdirectory")
    parser.add_argument("--output", type=Path, help="Override output manifest path")
    parser.add_argument("--overwrite", action="store_true", help="Replace a locally generated manifest")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.name):
        parser.error("--name must contain only letters, numbers, underscores or hyphens")
    destination = args.output or ROOT / "datasets" / args.name / "manifest.jsonl"
    index_dataset(args.source, destination, args.name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
