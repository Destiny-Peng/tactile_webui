#!/usr/bin/env python3
"""Measure lossy WebP compression of Deform images in one original episode SQLite.

Read-only source: episode.sqlite3, tables events/tactile_frames.
Pillow with WebP support is required: python -m pip install Pillow
Results: report.md, results.json, comparison.png in outputs/tactile_webp_benchmark/.
The temporary Q80 SQLite cache is removed unless --keep-cache is specified.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import sqlite3
import statistics
import time
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageDraw, ImageStat, features
except ImportError as exc:
    raise SystemExit("Install Pillow first: python -m pip install Pillow") from exc

ROOT = Path(__file__).resolve().parents[1]
QUALITIES = (60, 70, 80, 90)
QUERY = """
SELECT e.id, t.rowid, t.finger, t.deform_shape_json, t.deform_blob
FROM tactile_frames t JOIN events e ON e.id=t.event_id
WHERE e.source LIKE 'tactile:right:%' ORDER BY e.id
"""


def pct(values, fraction):
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * fraction
    i = int(pos)
    return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (pos - i)


def ms_stats(values):
    return {"n": len(values), "p50": pct(values, .5), "p95": pct(values, .95)}


def mib(value):
    return f"{value / 1048576:.2f} MiB"


def image_from_blob(event_id, shape_json, data):
    shape = json.loads(shape_json)
    if not isinstance(shape, list) or len(shape) not in (2, 3):
        raise ValueError(f"event {event_id}: invalid Deform shape {shape}")
    if len(shape) == 3 and shape[2] != 1:
        raise ValueError(f"event {event_id}: expected grayscale, got {shape}")
    height, width = (int(shape[0]), int(shape[1]))
    if height < 1 or width < 1 or data is None or len(data) != height * width:
        raise ValueError(f"event {event_id}: BLOB size disagrees with {shape}")
    return Image.frombytes("L", (width, height), data)


def compress(image, fmt, quality=None):
    buffer = io.BytesIO()
    if fmt == "PNG":
        image.save(buffer, format="PNG", compress_level=3)
    else:
        image.save(buffer, format="WEBP", quality=quality, method=4, lossless=False)
    return buffer.getvalue()


def quality_metrics(original, data):
    start = time.perf_counter_ns()
    with Image.open(io.BytesIO(data)) as decoded:
        decoded.load()
        recovered = decoded.convert("L")
    decode_ms = (time.perf_counter_ns() - start) / 1e6
    rms = ImageStat.Stat(ImageChops.difference(original, recovered)).rms[0]
    psnr = 100.0 if not rms else 20 * math.log10(255 / rms)
    return decode_ms, psnr


def read_latency(db, statement, keys):
    cursor = db.cursor()
    for key in keys[:20]:
        if cursor.execute(statement, (key,)).fetchone() is None:
            raise ValueError(f"missing event/rowid {key}")
    timings = []
    for key in keys:
        start = time.perf_counter_ns()
        if cursor.execute(statement, (key,)).fetchone() is None:
            raise ValueError(f"missing event/rowid {key}")
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return ms_stats(timings)


def preview(samples, output):
    if not samples:
        return
    originals = [image_from_blob(*x) for x in samples]
    width = max(im.width for im in originals)
    height = max(im.height for im in originals)
    gap, heading = 8, 28
    canvas = Image.new(
        "RGB", ((width + gap) * 5 + gap, (height + heading + gap) * len(samples) + gap), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for r, (sample, original) in enumerate(zip(samples, originals)):
        event_id = sample[0]
        for c, q in enumerate((None, *QUALITIES)):
            frame = original if q is None else Image.open(
                io.BytesIO(compress(original, "WEBP", q))
            ).convert("L")
            x = gap + c * (width + gap)
            y = gap + r * (height + heading + gap)
            draw.text((x, y + 5), f"event {event_id} | " + ("raw" if q is None else f"Q{q}"), fill="black")
            canvas.paste(frame.convert("RGB"), (x, y + heading))
    canvas.save(output / "comparison.png")


def run(args):
    if not features.check("webp"):
        raise RuntimeError("Pillow was built without WebP support")
    source = args.path.expanduser().resolve()
    if source.is_dir():
        source /= "episode.sqlite3"
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.max_images < 0 or args.quality_every < 1 or args.random_reads < 0:
        raise ValueError("invalid max-images / quality-every / random-reads")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = re.sub(r"[^\w.-]", "_", source.parent.name)[:60]
    output = (args.output or ROOT / "outputs" / "tactile_webp_benchmark" / f"{safe_name}_{stamp}").expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "cache_q80.sqlite3"

    labels = ["PNG", *(f"Q{q}" for q in QUALITIES)]
    byte_counts = {key: 0 for key in labels}
    encode_ms = {key: [] for key in labels}
    decode_ms = {key: [] for key in labels[1:]}
    psnr_scores = {key: [] for key in labels[1:]}
    fingers = Counter()
    examples, lookup_keys, original_bytes = [], [], 0
    rng = random.Random(42)
    began = time.perf_counter()
    print(f"[deform-webp] source (read-only): {source}", flush=True)
    print(f"[deform-webp] output: {output}", flush=True)
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as origin, closing(sqlite3.connect(cache_path)) as cache:
            origin.execute("PRAGMA query_only=ON")
            cache.execute("CREATE TABLE images (event_id INTEGER PRIMARY KEY, finger TEXT NOT NULL, webp BLOB NOT NULL)")
            cache.execute("BEGIN")
            count = 0
            for event_id, rowid, finger, shape, blob in origin.execute(QUERY):
                image = image_from_blob(event_id, shape, blob)
                count += 1
                original_bytes += len(blob)
                fingers[str(finger)] += 1
                lookup_keys.append((int(rowid), int(event_id)))
                candidate = (int(event_id), shape, blob)
                if len(examples) < 3:
                    examples.append(candidate)
                else:
                    slot = rng.randrange(count)
                    if slot < 3:
                        examples[slot] = candidate
                for key in labels:
                    started = time.perf_counter_ns()
                    compressed = compress(image, "PNG" if key == "PNG" else "WEBP",
                                          None if key == "PNG" else int(key[1:]))
                    encode_ms[key].append((time.perf_counter_ns() - started) / 1e6)
                    byte_counts[key] += len(compressed)
                    if key != "PNG" and (count == 1 or count % args.quality_every == 0):
                        dec_ms, score = quality_metrics(image, compressed)
                        decode_ms[key].append(dec_ms)
                        psnr_scores[key].append(score)
                    if key == "Q80":
                        cache.execute("INSERT INTO images VALUES (?, ?, ?)", (event_id, finger, compressed))
                if count % 1000 == 0:
                    cache.commit()
                    cache.execute("BEGIN")
                if count % 500 == 0:
                    print(f"[deform-webp] {count:,} images; raw {mib(original_bytes)}; Q80 {mib(byte_counts['Q80'])}", flush=True)
                if args.max_images and count >= args.max_images:
                    break
            if count == 0:
                raise ValueError("No right-hand tactile Deform images in events JOIN tactile_frames")
            cache.commit()
            cache.execute("VACUUM")
            cache_bytes = cache_path.stat().st_size
            scan_s = time.perf_counter() - began
            picks = random.Random(7).sample(lookup_keys, min(args.random_reads, len(lookup_keys)))
            raw_read = read_latency(origin, "SELECT deform_blob FROM tactile_frames WHERE rowid=?", [p[0] for p in picks])
            cache_read = read_latency(cache, "SELECT webp FROM images WHERE event_id=?", [p[1] for p in picks])

        preview(examples, output)
        metrics = {}
        for key in labels:
            item = {
                "compressed_bytes": byte_counts[key],
                "ratio_vs_raw": original_bytes / byte_counts[key],
                "reduction_vs_raw_pct": (1 - byte_counts[key] / original_bytes) * 100,
                "encode_ms": ms_stats(encode_ms[key]),
            }
            if key != "PNG":
                item["reduction_vs_png_pct"] = (1 - byte_counts[key] / byte_counts["PNG"]) * 100
                item["decode_ms"] = ms_stats(decode_ms[key])
                item["psnr_db"] = {"n": len(psnr_scores[key]), "p50": pct(psnr_scores[key], .5),
                                   "p05": pct(psnr_scores[key], .05)}
            metrics[key] = item
        result = {
            "source": str(source), "original_sqlite_bytes": source.stat().st_size,
            "tested_images": count, "max_images": args.max_images, "finger_counts": dict(fingers),
            "original_deform_bytes": original_bytes, "qualities": metrics,
            "compression_plus_cache_seconds": scan_s,
            "q80_cache_sqlite_bytes": cache_bytes,
            "q80_cache_sqlite_overhead_bytes": cache_bytes - byte_counts["Q80"],
            "raw_rowid_read_ms": raw_read, "q80_cache_event_id_read_ms": cache_read,
            "cache_retained": args.keep_cache, "quality_sample_every": args.quality_every,
            "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        rows = [
            "# Deform WebP compression benchmark", "",
            f"- Original SQLite (read-only): `{source}`",
            f"- Tested: **{count:,}** right-hand Deform images " + ("(whole episode)" if not args.max_images else "(capped sample; not a full-episode estimate)"),
            f"- Original Deform BLOB payload: **{mib(original_bytes)}**; entire source SQLite: {mib(source.stat().st_size)}",
            f"- Fingers: {dict(fingers)}", "",
            "| Format | Total payload | vs raw | vs PNG | Encode P50/P95 ms | Decode P50/P95 ms | PSNR P50/P05 dB |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key, item in metrics.items():
            dec = item.get("decode_ms", {})
            ps = item.get("psnr_db", {})
            versus = "—" if key == "PNG" else f"{item['reduction_vs_png_pct']:.1f}% smaller"
            rows.append(
                f"| {key} | {mib(item['compressed_bytes'])} | {item['ratio_vs_raw']:.2f}:1 | {versus} | "
                f"{item['encode_ms']['p50']:.2f}/{item['encode_ms']['p95']:.2f} | "
                + ("—" if key == "PNG" else f"{dec['p50']:.2f}/{dec['p95']:.2f}") + " | "
                + ("lossless" if key == "PNG" else f"{ps['p50']:.2f}/{ps['p05']:.2f}") + " |"
            )
        rows.extend([
            "", "## Q80 compressed SQLite cache (measured, not estimated)", "",
            f"- Cache file after VACUUM: **{mib(cache_bytes)}**, including event_id/finger indexes",
            f"- Cache overhead over WebP payload: {mib(cache_bytes - byte_counts['Q80'])}",
            f"- Whole benchmark (all qualities + PNG + cache): {scan_s:.2f} s",
            f"- Raw SQLite rowid lookup: P50 {raw_read['p50']:.3f} ms / P95 {raw_read['p95']:.3f} ms",
            f"- Q80 cache event_id lookup: P50 {cache_read['p50']:.3f} ms / P95 {cache_read['p95']:.3f} ms",
            f"- Queries: {raw_read['n']} each (warm-cache, random order)",
            "", "## Interpretation", "",
            "- Results cover Deform BLOB payloads, **not** all F6/Pose/video data in the source SQLite.",
            "- Raw access uses SQLite rowid (best-case indexed access, event_id→rowid mapping assumed).",
            "- Decode time and PSNR use every Nth image; identical reconstruction is capped at 100 dB PSNR.",
            "- Latencies exclude HTTP/browser/video decoding and cold-disk seeks.",
            "- Input SQLite was opened read-only; only outputs/ is written.",
            f"- Temporary Q80 cache {'kept (--keep-cache)' if args.keep_cache else 'removed after testing'}.",
            "", "## Files", "",
            "- `report.md` — copy this back to ChatGPT",
            "- `results.json` — detailed numbers",
            "- `comparison.png` — visual reference (raw, Q60, Q70, Q80, Q90)", ""
        ])
        (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = "\n".join(rows)
        (output / "report.md").write_text(report, encoding="utf-8")
        print("\n" + report)
        print(f"[deform-webp] Copy report: {output / 'report.md'}", flush=True)
        return output
    finally:
        if not args.keep_cache:
            cache_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to episode.sqlite3 or its episode directory")
    parser.add_argument("--output", type=Path, help="Empty output directory (default: outputs/tactile_webp_benchmark/...)")
    parser.add_argument("--max-images", type=int, default=0, help="Quick limited test; 0 (default) = whole episode")
    parser.add_argument("--quality-every", type=int, default=100, help="Measure PSNR/decode latency every N images")
    parser.add_argument("--random-reads", type=int, default=200, help="Warm-cache random SQLite reads")
    parser.add_argument("--keep-cache", action="store_true", help="Keep the tested compressed Q80 SQLite file")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        parser.exit(1, f"[deform-webp] ERROR: {exc}\n")


if __name__ == "__main__":
    main()
