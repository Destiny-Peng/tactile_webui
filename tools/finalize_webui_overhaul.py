from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


server = ROOT / "webui/server.py"
replace_once(
    server,
    '''EVENT_TYPES: dict[int, dict[str, str]] = {\n    6: {"name": "align_failure", "label": "Align failure", "outcome": "failure", "phase": "align"},\n    7: {"name": "insert_failure", "label": "Insert failure", "outcome": "failure", "phase": "insert"},\n    8: {"name": "align_success", "label": "Align success", "outcome": "success", "phase": "align"},\n    9: {"name": "insert_success", "label": "Insert success", "outcome": "success", "phase": "insert"},\n}\n''',
    '''LABEL_KEYS = (6, 7, 8, 9)\n''',
    "numeric label keys",
)
replace_once(
    server,
    '''            if rollout_id and key in EVENT_TYPES:\n                item = dict(row)\n                item["event_key"] = key\n                item.setdefault("event_name", EVENT_TYPES[key]["name"])\n                grouped[rollout_id].append(item)\n''',
    '''            if rollout_id and key in LABEL_KEYS:\n                item = dict(row)\n                item["event_key"] = key\n                item.pop("event_name", None)\n                grouped[rollout_id].append(item)\n''',
    "annotation read labels",
)
replace_once(server, "            if key not in EVENT_TYPES:\n", "            if key not in LABEL_KEYS:\n", "annotation validation")
replace_once(server, '''                    "event_key": key,\n                    "event_name": EVENT_TYPES[key]["name"],\n                    "start_frame": start,\n''', '''                    "event_key": key,\n                    "start_frame": start,\n''', "saved annotation semantics")
replace_once(server, '''            "event_types": EVENT_TYPES,\n''', '''            "event_types": list(LABEL_KEYS),\n''', "rollout event types")
replace_once(
    server,
    '''        for key, spec in EVENT_TYPES.items():\n            durations = duration_by_key[key]\n            type_rows.append(\n                {\n                    "event_key": key,\n                    **spec,\n                    "count": counter[key],\n                    "mean_duration_frames": (sum(durations) / len(durations)) if durations else None,\n                }\n            )\n''',
    '''        for key in LABEL_KEYS:\n            durations = duration_by_key[key]\n            type_rows.append(\n                {\n                    "event_key": key,\n                    "count": counter[key],\n                    "mean_duration_frames": (sum(durations) / len(durations)) if durations else None,\n                }\n            )\n''',
    "analysis numeric labels",
)
replace_once(server, '''                self.json_response(HTTPStatus.OK, {"event_types": EVENT_TYPES})\n''', '''                self.json_response(HTTPStatus.OK, {"event_types": list(LABEL_KEYS)})\n''', "event types endpoint")
replace_once(
    server,
    '''        except self.CLIENT_DISCONNECT_ERRORS:\n            return False\n\n    def _read_json_body''',
    '''        except self.CLIENT_DISCONNECT_ERRORS:\n            self._client_disconnected = True\n            return False\n\n    def _read_json_body''',
    "header disconnect monitoring",
)

workspace_test = ROOT / "webui/tests/test_workspace.py"
replace_once(
    workspace_test,
    '''        self.assertEqual(\n            [row["event_name"] for row in saved],\n            ["align_failure", "insert_failure", "align_success", "insert_success"],\n        )\n''',
    '''        self.assertTrue(all("event_name" not in row for row in saved))\n''',
    "annotation test semantics",
)

css = ROOT / "webui/static/shell.css"
replace_once(
    css,
    '''.precision-timeline { position: relative; min-height: 182px; padding: 23px 10px 23px 44px; overflow: hidden; background: linear-gradient(180deg,#fbfcfe,#f7f9fc); border: 1px solid var(--line-soft); border-radius: 9px; user-select: none; }\n.precision-timeline.compact-timeline { min-height: 142px; }\n.timeline-track-area { position: relative; min-height: 134px; margin-left: 34px; }\n.compact-timeline .timeline-track-area { min-height: 96px; }\n''',
    '''.precision-timeline { position: relative; min-height: 184px; padding: 17px 10px 10px; overflow: hidden; background: linear-gradient(180deg,#fbfcfe,#f7f9fc); border: 1px solid var(--line-soft); border-radius: 9px; user-select: none; }\n.precision-timeline.compact-timeline { min-height: 152px; }\n.timeline-track-area { position: relative; min-height: 133px; margin: 0 0 20px 34px; }\n.compact-timeline .timeline-track-area { min-height: 133px; }\n''',
    "timeline geometry",
)
replace_once(css, '''.timeline-axis-label { position: absolute; bottom: 5px; color: #8290a1; font: .55rem/1 ui-monospace,monospace; transform: translateX(-50%); pointer-events: none; }\n''', '''.timeline-axis-label { position: absolute; bottom: -17px; color: #8290a1; font: .55rem/1 ui-monospace,monospace; transform: translateX(-50%); pointer-events: none; }\n''', "timeline axis")
replace_once(css, '''.timeline-playhead { position: absolute; z-index: 9; top: 8px; bottom: 18px; width: 2px; background: #265ee8;''', '''.timeline-playhead { position: absolute; z-index: 9; top: 0; bottom: 0; width: 2px; background: #265ee8;''', "timeline playhead")

js = ROOT / "webui/static/shell.js"
replace_once(
    js,
    '''    row.addEventListener("click", function () { state.activeEvent = index; renderIntervals(); renderAnnotateTimeline(); });\n    row.querySelector(".event-type").addEventListener("change", function (event) { event.stopPropagation(); state.events[index].event_key = Number(event.target.value); markDirty(); renderAnnotateTimeline(); });\n    row.querySelector(".event-start").addEventListener("change", function (event) { event.stopPropagation(); state.events[index].start_frame = clampFrame(event.target.value); if (state.events[index].end_frame < state.events[index].start_frame) state.events[index].end_frame = state.events[index].start_frame; markDirty(); renderIntervals(); renderAnnotateTimeline(); });\n    row.querySelector(".event-end").addEventListener("change", function (event) { event.stopPropagation(); state.events[index].end_frame = Math.max(state.events[index].start_frame, clampFrame(event.target.value)); markDirty(); renderIntervals(); renderAnnotateTimeline(); });\n''',
    '''    row.addEventListener("click", function (event) { if (event.target.closest("input,select,button")) return; state.activeEvent = index; renderIntervals(); renderAnnotateTimeline(); });\n    row.addEventListener("focusin", function () { state.activeEvent = index; container.querySelectorAll(".interval-row").forEach(function (item) { item.classList.toggle("active", Number(item.dataset.eventIndex) === index); }); renderAnnotateTimeline(); });\n    row.querySelector(".event-type").addEventListener("change", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].event_key = Number(event.target.value); markDirty(); renderAnnotateTimeline(); });\n    row.querySelector(".event-start").addEventListener("change", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].start_frame = clampFrame(event.target.value); if (state.events[index].end_frame < state.events[index].start_frame) state.events[index].end_frame = state.events[index].start_frame; markDirty(); renderIntervals(); renderAnnotateTimeline(); });\n    row.querySelector(".event-end").addEventListener("change", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].end_frame = Math.max(state.events[index].start_frame, clampFrame(event.target.value)); markDirty(); renderIntervals(); renderAnnotateTimeline(); });\n''',
    "interval control focus",
)
replace_once(
    js,
    '''    var data = await jsonRequest("/api/diagnostics"); var latency = data.latency_ms || {}; var errors = (data.server_errors || []).length + (data.client_errors || []).length;\n    byId("diagnosticKpis").innerHTML = [['Requests',data.requests || 0],['p50',latency.p50 == null ? '—' : Math.round(latency.p50) + ' ms'],['p95',latency.p95 == null ? '—' : Math.round(latency.p95) + ' ms'],['Errors',errors],['Uptime',Math.round((data.uptime_seconds || 0)/60) + ' min'],['Video req',(data.resource_counts || {}).video || 0]].map(function (item) { return '<div class="diagnostic-kpi"><span class="eyebrow">' + escapeHtml(item[0]) + '</span><strong>' + escapeHtml(item[1]) + '</strong></div>'; }).join("");\n''',
    '''    var data = await jsonRequest("/api/diagnostics"); var videoLatency = (data.latency_by_resource_ms || {}).video || {}; var errors = (data.server_errors || []).length + (data.client_errors || []).length;\n    byId("diagnosticKpis").innerHTML = [['Requests',data.requests || 0],['Video p50',videoLatency.p50 == null ? '—' : Math.round(videoLatency.p50) + ' ms'],['Video p95',videoLatency.p95 == null ? '—' : Math.round(videoLatency.p95) + ' ms'],['Errors',errors],['Uptime',Math.round((data.uptime_seconds || 0)/60) + ' min'],['Video req',(data.resource_counts || {}).video || 0]].map(function (item) { return '<div class="diagnostic-kpi"><span class="eyebrow">' + escapeHtml(item[0]) + '</span><strong>' + escapeHtml(item[1]) + '</strong></div>'; }).join("");\n''',
    "video diagnostics latency",
)
replace_once(
    js,
    '''function initPerformanceObserver() {\n  if (!("PerformanceObserver" in window)) return;\n  try {\n    var observer = new PerformanceObserver(function (list) {\n      list.getEntries().forEach(function (entry) {\n        if (entry.name.indexOf("/api/videos/") < 0 && entry.name.indexOf("/api/tactile/") < 0) return;\n        queueTelemetry({ event: "resource_timing", level: "info", url: entry.name, duration_ms: entry.duration, transfer_size: entry.transferSize || 0, encoded_size: entry.encodedBodySize || 0, decoded_size: entry.decodedBodySize || 0 });\n      });\n    }); observer.observe({ type: "resource", buffered: true });\n  } catch (_error) {}\n}\n''',
    '''function initPerformanceObserver() {\n  if (!("PerformanceObserver" in window)) return;\n  var tactileCounter = 0;\n  try {\n    var observer = new PerformanceObserver(function (list) {\n      list.getEntries().forEach(function (entry) {\n        var isVideo = entry.name.indexOf("/api/videos/") >= 0;\n        var isTactile = entry.name.indexOf("/api/tactile/") >= 0;\n        if (!isVideo && !isTactile) return;\n        if (isTactile) { tactileCounter += 1; if (entry.duration < 250 && tactileCounter % 30 !== 0) return; }\n        queueTelemetry({ event: "resource_timing", level: entry.duration >= 1000 ? "error" : "info", resource: isVideo ? "video" : "tactile", url: entry.name, duration_ms: entry.duration, transfer_size: entry.transferSize || 0, encoded_size: entry.encodedBodySize || 0, decoded_size: entry.decodedBodySize || 0 });\n      });\n    }); observer.observe({ type: "resource", buffered: true });\n  } catch (_error) {}\n}\n''',
    "resource sampling",
)

readme = ROOT / "README.md"
replace_once(
    readme,
    '''The detailed Results viewer is also directly available at `http://127.0.0.1:8765/tactile`.\n\n## Annotation schema\n\nThe standalone annotator edits only the four USB tactile interval types already consumed by `tools/sharpa_tactile`:\n\n| event_key | label | outcome |\n|---:|---|---|\n| 6 | Align failure | failure |\n| 7 | Insert failure | failure |\n| 8 | Align success | success |\n| 9 | Insert success | success |\n\nEach saved record keeps the experiment-compatible fields `rollout_id`, `event_index`, `event_key`, `start_frame`, and `end_frame`.\n''',
    '''The synchronized tactile diagnostic viewer remains directly available at `http://127.0.0.1:8765/tactile`. The main **Results** page is now the online-detection review surface.\n\n## Annotation schema\n\nThe standalone annotator intentionally exposes only numeric labels **6, 7, 8, 9**. Semantic label names are not stored in new tactile sidecars or shown in the annotation UI.\n\nEach saved record keeps the experiment-compatible fields `rollout_id`, `event_index`, `event_key`, `start_frame`, and `end_frame`.\n''',
    "README numeric labels",
)
replace_once(
    readme,
    '''## Tactile Results viewer\n\nResults keeps the optimized synchronized path from the dissertation WebUI:\n\n- synchronized robot video;\n- five-finger raw/deform tactile sprites;\n- F6 values/history;\n- one five-finger sprite per synchronized frame;\n- small forward sprite prefetch window;\n- de-duplicated tactile-series requests.\n''',
    '''## Results and diagnostics\n\nThe main Results page reviews online detection against the robot video on one shared frame axis. It shows saved numeric annotation intervals, numeric GT labels, numeric predicted labels, and the three model probability curves (`p0`, `p1`, `p2`). Existing SHARPA `test_predictions.csv` files below the configured Runs root are discovered automatically.\n\nThe legacy `/tactile` route remains a low-level synchronized tactile diagnostic viewer.\n\nWebUI stability/performance telemetry is written only below:\n\n```text\nlogs/webui/server/\nlogs/webui/client/\n```\n\nThe Settings → Diagnostics panel reports server/video latency, HTTP errors, browser video stalls/errors and the active log paths. Successful high-frequency tactile requests are sampled so monitoring does not add material I/O load.\n''',
    "README results diagnostics",
)
