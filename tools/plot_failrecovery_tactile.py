#!/usr/bin/env python3
"""Compare thumb, index and middle F/M magnitudes for success and failure."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle")
COLORS = {"success": "#2474B5", "failure": "#D96B35"}


def read_series(record: dict) -> tuple[dict, dict]:
    grouped = {finger: [] for finger in FINGERS}
    invalid = {finger: 0 for finger in FINGERS}
    with (PROJECT_ROOT / record["tactile_events_path"]).open() as handle:
        for line in handle:
            event = json.loads(line)
            finger = event["finger"]
            if finger not in grouped:
                continue
            f6 = event.get("f6")
            values = np.full(6, np.nan)
            if event.get("valid") and isinstance(f6, list) and len(f6) == 6:
                values = np.asarray(f6, dtype=float)
            else:
                invalid[finger] += 1
            grouped[finger].append((int(event["receive_mono_ns"]), values))
    origin = min(timestamp for points in grouped.values() for timestamp, _ in points)
    series = {}
    for finger, points in grouped.items():
        points.sort(key=lambda point: point[0])
        series[finger] = (
            np.asarray([(timestamp - origin) / 1e9 for timestamp, _ in points]),
            np.asarray([values for _, values in points]),
        )
    return series, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl")
    parser.add_argument("--task-key", default="usb_socket")
    parser.add_argument("--min-y-span", type=float, default=1.0,
                        help="Minimum F magnitude axis span (default: 1).")
    parser.add_argument("--min-m-span", type=float, default=0.01,
                        help="Minimum M magnitude axis span (default: 0.01).")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/realrobot/tactile_plots" / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    for name, value in (("--min-y-span", args.min_y_span), ("--min-m-span", args.min_m_span)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = {}
    for outcome in ("success", "failure"):
        candidates = [r for r in records if r["task_key"] == args.task_key and r["ground_truth_outcome"] == outcome and not r.get("quality_flags")]
        if not candidates:
            raise ValueError(f"No {outcome} rollout for {args.task_key}")
        reviewed = [r for r in candidates if r.get("outcome_source") == "review.json"]
        selected[outcome] = (reviewed or candidates)[0]
    histories = {outcome: read_series(record) for outcome, record in selected.items()}
    magnitudes = {
        outcome: {
            finger: {
                "F": np.linalg.norm(series[finger][1][:, :3], axis=1),
                "M": np.linalg.norm(series[finger][1][:, 3:6], axis=1),
            }
            for finger in FINGERS
        }
        for outcome, (series, _) in histories.items()
    }
    # Both outcomes share each subplot; F and M have independent physical scales.
    y_limits = {}
    for finger in FINGERS:
        y_limits[finger] = {}
        for component, minimum in (("F", args.min_y_span), ("M", args.min_m_span)):
            values = np.concatenate([magnitudes[outcome][finger][component] for outcome in selected])
            finite = values[np.isfinite(values)]
            peak = float(finite.max()) if finite.size else 0.0
            y_limits[finger][component] = (0.0, max(minimum, peak * 1.08))
    max_time = max(float(series[finger][0][-1]) for series, _ in histories.values() for finger in FINGERS)
    x_ticks = MaxNLocator(nbins=6).tick_values(0.0, max(max_time, 1.0))
    x_limits = (0.0, float(x_ticks[-1]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task_key": args.task_key,
        "plotted_fingers": list(FINGERS),
        "time_axis": "receive_mono_ns relative to first tactile event, seconds",
        "values": "F = sqrt(sum(f6[0:3]**2)); M = sqrt(sum(f6[3:6]**2)); source units unspecified; no smoothing or normalization",
        "invalid_samples": "shown as gaps",
        "shared_axis_limits": {"x_seconds": x_limits, "y_magnitude_by_finger": y_limits},
        "min_y_span": {"F": args.min_y_span, "M": args.min_m_span},
        "rollouts": {},
    }
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": "#374151", "text.color": "#1F2937", "axes.edgecolor": "#CBD5E1"})
    for outcome, record in selected.items():
        series, invalid = histories[outcome]
        outcome_label = "Success" if outcome == "success" else "Fail"
        fig, axes = plt.subplots(len(FINGERS), 2, figsize=(13, 7.5), sharex=True)
        for row, finger in enumerate(FINGERS):
            for column, component in enumerate(("F", "M")):
                ax = axes[row, column]
                times = histories[outcome][0][finger][0]
                ax.plot(times, magnitudes[outcome][finger][component],
                        color=COLORS[outcome], linewidth=1.5, alpha=0.95,
                        label=outcome_label)
                if row == 0:
                    ax.set_title(r"Force magnitude $\|F\|_2$" if component == "F" else r"Moment magnitude $\|M\|_2$", fontsize=13, pad=14)
                if column == 0:
                    ax.set_ylabel(finger.capitalize(), fontsize=11, labelpad=10)
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits[finger][component])
                ax.set_xticks(x_ticks)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(axis="both", labelsize=9, length=3, color="#CBD5E1")
        for ax in axes[-1]:
            ax.set_xlabel("Elapsed time (s)", labelpad=9)
        fig.suptitle(record["task_description"], fontsize=16, fontweight="semibold", y=0.985)
        fig.text(0.5, 0.945, f"{outcome_label} | Rollout ID: {record['id']}", ha="center", fontsize=9, color=COLORS[outcome])
        fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=1.2, w_pad=2.5)
        prefix = args.output_root / f"{outcome}_three_finger_fm_norms"
        fig.savefig(prefix.with_suffix(".png"), dpi=200, facecolor="white")
        fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        metadata["rollouts"][outcome] = {
            "id": record["id"], "episode_label": record["episode_label"],
            "task_description": record["task_description"],
            "camera_video_paths": record["camera_video_paths"],
            "outcome_source": record.get("outcome_source"),
            "tactile_events_path": record["tactile_events_path"],
            "sample_counts": {finger: len(series[finger][0]) for finger in FINGERS},
            "invalid_sample_counts": invalid,
            "png": str(prefix.with_suffix(".png").relative_to(PROJECT_ROOT)),
            "pdf": str(prefix.with_suffix(".pdf").relative_to(PROJECT_ROOT)),
        }
    (args.output_root / "fm_norms_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
