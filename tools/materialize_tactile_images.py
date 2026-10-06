#!/usr/bin/env python3
"""Materialize exported tactile byte streams as static PNG images.

Images are grouped by kind and finger so episode roots stay compact:

    tactile/images/deform/index/000123.png
    tactile/images/raw/index/000123.png

The filename is the per-finger sample_index from tactile/events.jsonl. Existing
files are reused unless --overwrite is supplied. A symlinked standalone
manifest is supported: paths stored in the manifest are resolved against the
original project tree, without copying the dataset payload.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
import zlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
KINDS = ("raw", "deform")


def _project_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing project-relative path")
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    path.relative_to(root)
    return path


def _manifest_references(manifest: Path, limit: int = 16) -> list[Path]:
    references: list[Path] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            cameras = row.get("camera_video_paths")
            if isinstance(cameras, dict):
                for value in cameras.values():
                    if isinstance(value, str) and value.strip() and not Path(value).expanduser().is_absolute():
                        references.append(Path(value))
            for key in ("synchronized_frames_path", "tactile_events_path"):
                value = row.get(key)
                if isinstance(value, str) and value.strip() and not Path(value).expanduser().is_absolute():
                    references.append(Path(value))
            streams = row.get("tactile_stream_paths")
            if isinstance(streams, dict):
                for entry in streams.values():
                    if not isinstance(entry, dict):
                        continue
                    for value in entry.values():
                        if isinstance(value, str) and value.strip() and not Path(value).expanduser().is_absolute():
                            references.append(Path(value))
            if len(references) >= limit:
                break
    return references[:limit]


def infer_data_root(manifest: Path, fallback: Path) -> Path:
    """Infer which project root the manifest's relative paths belong to."""
    fallback = fallback.resolve()
    manifest = manifest.resolve()
    try:
        references = _manifest_references(manifest)
    except (OSError, json.JSONDecodeError):
        return fallback
    if not references:
        return fallback

    candidates = [fallback, *manifest.parents]
    best_root = fallback
    best_score = -1
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        score = sum((candidate / value).exists() for value in references)
        if score > best_score:
            best_root = candidate
            best_score = score
        if score == len(references):
            break
    return best_root if best_score > 0 else fallback


def _stream_paths(root: Path, row: dict[str, Any]) -> dict[str, dict[str, Path]]:
    raw = row.get("tactile_stream_paths")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, Path]] = {}
    for finger in FINGERS:
        entry = raw.get(finger)
        found: dict[str, Path] = {}
        if isinstance(entry, dict):
            for kind in KINDS:
                value = entry.get(kind) or entry.get(kind + "_path") or entry.get(kind + "_u8")
                if value:
                    found[kind] = _project_file(root, value)
        for kind in KINDS:
            if kind in found:
                continue
            value = raw.get(finger + "_" + kind) or raw.get(finger + "." + kind) or raw.get(finger + "_" + kind + "_path")
            if value:
                found[kind] = _project_file(root, value)
        if found:
            result[finger] = found
    return result


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def png_bytes(data: bytes, shape: Any, compression_level: int = 3) -> bytes:
    dims = [int(value) for value in shape]
    if len(dims) == 2:
        height, width = dims
        channels = 1
    elif len(dims) == 3 and dims[2] in (1, 3, 4):
        height, width, channels = dims
    else:
        raise ValueError(f"unsupported tactile image shape: {shape}")
    if height <= 0 or width <= 0 or len(data) != height * width * channels:
        raise ValueError("invalid tactile image dimensions or byte length")
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    stride = width * channels
    scanlines = b"".join(b"\x00" + data[offset:offset + stride] for offset in range(0, len(data), stride))
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(scanlines, level=compression_level)) + _png_chunk(b"IEND", b"")


def image_path(stream_path: Path, kind: str, finger: str, sample_index: int) -> Path:
    return stream_path.parent / "images" / kind / finger / f"{sample_index:06d}.png"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        # These PNGs are derived and fully regenerable. Atomic replacement is
        # enough; fsync-per-image made large dataset preparation unnecessarily
        # slow by forcing hundreds of thousands of synchronous flushes.
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def materialize_rollout(root: Path, row: dict[str, Any], kinds: tuple[str, ...], overwrite: bool) -> dict[str, int]:
    rollout_id = str(row.get("id") or "").strip()
    events_path = _project_file(root, row.get("tactile_events_path"))
    streams = _stream_paths(root, row)
    counts = {kind: 0 for kind in kinds}
    skipped = 0
    with events_path.open("r", encoding="utf-8") as event_file, ExitStack() as stack:
        handles: dict[tuple[str, str], Any] = {}
        for finger, by_kind in streams.items():
            for kind, path in by_kind.items():
                if kind in kinds:
                    handles[(finger, kind)] = stack.enter_context(path.open("rb"))
        for line_number, line in enumerate(event_file, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            finger = str(event.get("finger") or "").strip().lower()
            if finger not in FINGERS:
                continue
            sample_index = int(event.get("sample_index"))
            for kind in kinds:
                handle = handles.get((finger, kind))
                stream_path = streams.get(finger, {}).get(kind)
                if handle is None or stream_path is None:
                    continue
                offset = event.get(kind + "_offset_bytes")
                length = event.get(kind + "_length_bytes")
                shape = event.get(kind + "_shape")
                if offset is None or length is None or shape is None:
                    continue
                target = image_path(stream_path, kind, finger, sample_index)
                if target.is_file() and not overwrite:
                    skipped += 1
                    continue
                offset = int(offset)
                length = int(length)
                if offset < 0 or length <= 0:
                    raise ValueError(f"{rollout_id}: invalid {kind} byte range at events line {line_number}")
                handle.seek(offset)
                data = handle.read(length)
                if len(data) != length:
                    raise ValueError(f"{rollout_id}: short {kind} read at events line {line_number}")
                _atomic_bytes(target, png_bytes(data, shape, compression_level=3))
                counts[kind] += 1
    counts["skipped"] = skipped
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", default="datasets/failrecovery/manifest.jsonl")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    manifest_entry = Path(args.manifest).expanduser()
    if not manifest_entry.is_absolute():
        manifest_entry = root / manifest_entry
    if not manifest_entry.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_entry}")
    manifest = manifest_entry.resolve()
    data_root = infer_data_root(manifest, root)
    kinds = tuple(dict.fromkeys(args.kinds))
    totals = {kind: 0 for kind in kinds}
    totals["skipped"] = 0
    rollouts = 0
    print(f"[tactile-images] manifest={manifest}")
    print(f"[tactile-images] data_root={data_root}")
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("tactile_events_path"):
                continue
            summary = materialize_rollout(data_root, row, kinds, args.overwrite)
            rollouts += 1
            for key, value in summary.items():
                totals[key] = totals.get(key, 0) + value
            print(f"[tactile-images] {row.get('id')}: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
    print(f"[tactile-images] complete: rollouts={rollouts}, " + ", ".join(f"{key}={value}" for key, value in totals.items()))


if __name__ == "__main__":
    main()
