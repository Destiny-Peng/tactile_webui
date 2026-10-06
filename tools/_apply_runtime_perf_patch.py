from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Server: persistent HTTP/1.1 connections, per-request monitoring, one-pass run scan.
# ---------------------------------------------------------------------------
server = ROOT / "webui/server.py"
replace_once(
    server,
    '    protocol_version = "HTTP/1.0"\n',
    '    protocol_version = "HTTP/1.1"\n',
    "HTTP/1.1",
)
replace_once(
    server,
    '''    def finish(self) -> None:\n        try:\n            super().finish()\n        finally:\n            started = getattr(self, "_request_started", None)\n            if started is None or getattr(self, "_request_recorded", False):\n                return\n            self._request_recorded = True\n            try:\n                self.app.monitor.record_request(\n                    method=getattr(self, "_request_method", "?"),\n                    path=getattr(self, "_request_path", "?"),\n                    status=getattr(self, "_request_status", 500),\n                    resource=getattr(self, "_request_resource", "other"),\n                    duration_ms=round((time.perf_counter() - started) * 1000.0, 3),\n                    response_bytes=getattr(self, "_response_bytes", None),\n                    client_disconnected=bool(getattr(self, "_client_disconnected", False)),\n                    range=getattr(self, "_video_range", None),\n                    video_total_bytes=getattr(self, "_video_total_bytes", None),\n                )\n            except OSError:\n                pass\n''',
    '''    def _record_request(self) -> None:\n        started = getattr(self, "_request_started", None)\n        if started is None or getattr(self, "_request_recorded", False):\n            return\n        self._request_recorded = True\n        try:\n            self.app.monitor.record_request(\n                method=getattr(self, "_request_method", "?"),\n                path=getattr(self, "_request_path", "?"),\n                status=getattr(self, "_request_status", 500),\n                resource=getattr(self, "_request_resource", "other"),\n                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),\n                response_bytes=getattr(self, "_response_bytes", None),\n                client_disconnected=bool(getattr(self, "_client_disconnected", False)),\n                range=getattr(self, "_video_range", None),\n                video_total_bytes=getattr(self, "_video_total_bytes", None),\n            )\n        except OSError:\n            pass\n\n    def finish(self) -> None:\n        try:\n            super().finish()\n        finally:\n            # Safety net for a request that escaped before do_GET/do_POST's\n            # finally block. With HTTP/1.1 this method runs per connection, not\n            # per request, so normal accounting is done in the request methods.\n            self._record_request()\n''',
    "per-request monitoring",
)
replace_once(
    server,
    '''    def _finish_headers(self) -> bool:\n        self.close_connection = True\n        try:\n            self.end_headers()\n            return True\n        except self.CLIENT_DISCONNECT_ERRORS:\n            self._client_disconnected = True\n            return False\n''',
    '''    def _finish_headers(self) -> bool:\n        try:\n            self.end_headers()\n            return True\n        except self.CLIENT_DISCONNECT_ERRORS:\n            self.close_connection = True\n            self._client_disconnected = True\n            return False\n''',
    "persistent headers",
)
text = server.read_text(encoding="utf-8")
connection_header = '        self.send_header("Connection", "close")\n'
if text.count(connection_header) < 4:
    raise SystemExit("expected Connection: close headers")
server.write_text(text.replace(connection_header, ""), encoding="utf-8")
replace_once(
    server,
    '        self.send_header("Cache-Control", "private, max-age=3600")\n',
    '        self.send_header("Cache-Control", "private, max-age=86400, immutable")\n',
    "long tactile image cache",
)
replace_once(
    server,
    '''        found: dict[Path, set[str]] = defaultdict(set)\n        for marker in markers:\n            for path in root.glob(f"**/{marker}"):\n                if not path.is_file():\n                    continue\n                relative = path.parent.relative_to(root)\n                if len(relative.parts) > 4:\n                    continue\n                found[path.parent].add(marker)\n''',
    '''        found: dict[Path, set[str]] = defaultdict(set)\n        # One bounded traversal instead of one recursive glob per marker.\n        for current, directories, filenames in os.walk(root):\n            directory = Path(current)\n            relative = directory.relative_to(root)\n            depth = len(relative.parts)\n            if depth > 4:\n                directories[:] = []\n                continue\n            matched = markers.intersection(filenames)\n            if matched:\n                found[directory].update(matched)\n            if depth >= 4:\n                directories[:] = []\n''',
    "single-pass run scan",
)
replace_once(
    server,
    '''        except OSError as exc:\n            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))\n\n    def do_POST(self) -> None:\n''',
    '''        except OSError as exc:\n            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))\n        finally:\n            self._record_request()\n\n    def do_POST(self) -> None:\n''',
    "GET request accounting",
)
replace_once(
    server,
    '''        except OSError as exc:\n            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))\n\n\ndef main() -> None:\n''',
    '''        except OSError as exc:\n            self.json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))\n        finally:\n            self._record_request()\n\n\ndef main() -> None:\n''',
    "POST request accounting",
)


# ---------------------------------------------------------------------------
# Annotate: cap tactile loads at one five-finger batch and coalesce to latest frame.
# ---------------------------------------------------------------------------
shell = ROOT / "webui/static/shell.js"
replace_once(
    shell,
    '''  tactileSeriesPromise: null,\n  tactileAppliedKey: "",\n  results: {''',
    '''  tactileSeriesPromise: null,\n  tactileAppliedKey: "",\n  tactileLoadActive: false,\n  tactilePendingFrame: null,\n  results: {''',
    "annotate tactile queue state",
)
replace_once(
    shell,
    '''  state.tactileSeriesPromise = null;\n  state.tactileAppliedKey = "";\n  byId("annotateEmpty").classList.add("hidden");''',
    '''  state.tactileSeriesPromise = null;\n  state.tactileAppliedKey = "";\n  state.tactilePendingFrame = null;\n  state.tactileGeneration += 1;\n  byId("annotateEmpty").classList.add("hidden");''',
    "annotate tactile epoch reset",
)
old_update = '''async function updateAnnotateTactile(frame) {\n  var record = selectedRollout();\n  if (!record) return;\n  var camera = byId("annotateCamera").value || chooseCamera(record);\n  var kind = state.settings && state.settings.default_tactile_kind || "deform";\n  var generation = ++state.tactileGeneration;\n  var series = await ensureAnnotateTactileSeries(record, camera);\n  if (generation !== state.tactileGeneration || !series) return;\n  var sync = nearestAnnotateTactileSync(series, frame);\n  if (!sync) { byId("annotateTactileStatus").textContent = "No synchronized tactile at this frame"; return; }\n  var fingerRows = sync.fingers || {};\n  var key = record.id + "|" + camera + "|" + sync.frame + "|" + kind + "|"\n    + TACTILE_FINGERS.map(function (finger) { return fingerRows[finger] && fingerRows[finger].event_id != null ? fingerRows[finger].event_id : "-"; }).join(",");\n  if (key === state.tactileAppliedKey) return;\n  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · loading " + kind;\n  var cells = [];\n  var loads = [];\n  TACTILE_FINGERS.forEach(function (finger) {\n    var info = fingerRows[finger] || {};\n    if (info.event_id == null) { cells.push({ finger: finger, missing: true }); return; }\n    var url = annotateTactileImageUrl(record, finger, info.event_id, kind);\n    cells.push({ finger: finger, url: url });\n    loads.push(new Promise(function (resolve, reject) {\n      var image = new Image();\n      image.onload = resolve;\n      image.onerror = function () { reject(new Error(finger + " tactile image failed")); };\n      image.src = url;\n    }));\n  });\n  try {\n    await Promise.all(loads);\n  } catch (error) {\n    if (generation !== state.tactileGeneration) return;\n    byId("annotateTactileStatus").textContent = "Tactile image load failed; keeping previous frame";\n    queueTelemetry({ event: "tactile_image_error", level: "error", rollout_id: record.id, camera: camera, frame: frame, matched_frame: sync.frame, kind: kind, message: String(error.message || error) });\n    return;\n  }\n  if (generation !== state.tactileGeneration) return;\n  byId("annotateTactileGrid").innerHTML = cells.map(function (cell) {\n    return '<div class="annotate-tactile-finger"><strong>' + escapeHtml(cell.finger) + '</strong>'\n      + (cell.missing ? '<div class="tactile-missing">No sample</div>' : '<img src="' + escapeHtml(cell.url) + '" alt="' + escapeHtml(cell.finger + " " + kind) + '">') + '</div>';\n  }).join("");\n  state.tactileAppliedKey = key;\n  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · " + kind;\n}\n'''
new_update = '''async function loadAnnotateTactileFrame(frame, generation) {\n  var record = selectedRollout();\n  if (!record) return;\n  var camera = byId("annotateCamera").value || chooseCamera(record);\n  var kind = state.settings && state.settings.default_tactile_kind || "deform";\n  var series = await ensureAnnotateTactileSeries(record, camera);\n  if (generation !== state.tactileGeneration || !series) return;\n  var sync = nearestAnnotateTactileSync(series, frame);\n  if (!sync) { byId("annotateTactileStatus").textContent = "No synchronized tactile at this frame"; return; }\n  var fingerRows = sync.fingers || {};\n  var key = record.id + "|" + camera + "|" + sync.frame + "|" + kind + "|"\n    + TACTILE_FINGERS.map(function (finger) { return fingerRows[finger] && fingerRows[finger].event_id != null ? fingerRows[finger].event_id : "-"; }).join(",");\n  if (key === state.tactileAppliedKey) return;\n  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · loading " + kind;\n  var cells = [];\n  var loads = [];\n  var started = performance.now();\n  TACTILE_FINGERS.forEach(function (finger) {\n    var info = fingerRows[finger] || {};\n    if (info.event_id == null) { cells.push({ finger: finger, missing: true }); return; }\n    var url = annotateTactileImageUrl(record, finger, info.event_id, kind);\n    cells.push({ finger: finger, url: url });\n    loads.push(new Promise(function (resolve, reject) {\n      var image = new Image();\n      image.onload = resolve;\n      image.onerror = function () { reject(new Error(finger + " tactile image failed")); };\n      image.src = url;\n    }));\n  });\n  try {\n    await Promise.all(loads);\n  } catch (error) {\n    if (generation !== state.tactileGeneration) return;\n    byId("annotateTactileStatus").textContent = "Tactile image load failed; keeping previous frame";\n    queueTelemetry({ event: "tactile_image_error", level: "error", rollout_id: record.id, camera: camera, frame: frame, matched_frame: sync.frame, kind: kind, message: String(error.message || error) });\n    return;\n  }\n  if (generation !== state.tactileGeneration || record.id !== state.selectedId) return;\n  byId("annotateTactileGrid").innerHTML = cells.map(function (cell) {\n    return '<div class="annotate-tactile-finger"><strong>' + escapeHtml(cell.finger) + '</strong>'\n      + (cell.missing ? '<div class="tactile-missing">No sample</div>' : '<img src="' + escapeHtml(cell.url) + '" alt="' + escapeHtml(cell.finger + " " + kind) + '">') + '</div>';\n  }).join("");\n  state.tactileAppliedKey = key;\n  var duration = performance.now() - started;\n  if (duration >= 250) queueTelemetry({ event: "tactile_batch_slow", level: duration >= 1000 ? "error" : "info", rollout_id: record.id, camera: camera, frame: frame, matched_frame: sync.frame, kind: kind, duration_ms: duration });\n  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · " + kind;\n}\nasync function drainAnnotateTactileQueue() {\n  if (state.tactileLoadActive) return;\n  state.tactileLoadActive = true;\n  var generation = state.tactileGeneration;\n  try {\n    while (state.tactilePendingFrame != null && generation === state.tactileGeneration) {\n      var frame = state.tactilePendingFrame;\n      state.tactilePendingFrame = null;\n      await loadAnnotateTactileFrame(frame, generation);\n    }\n  } finally {\n    state.tactileLoadActive = false;\n    if (state.tactilePendingFrame != null) Promise.resolve().then(drainAnnotateTactileQueue);\n  }\n}\nfunction updateAnnotateTactile(frame) {\n  state.tactilePendingFrame = clampFrame(frame);\n  if (!state.tactileLoadActive) drainAnnotateTactileQueue();\n}\n'''
replace_once(shell, old_update, new_update, "annotate tactile backpressure")
replace_once(
    shell,
    '''  byId("annotateCamera").addEventListener("change", function () { state.tactileSeries = null; state.tactileSeriesKey = ""; state.tactileSeriesPromise = null; state.tactileAppliedKey = ""; loadAnnotateVideo(); seekFrame(0); });''',
    '''  byId("annotateCamera").addEventListener("change", function () { state.tactileSeries = null; state.tactileSeriesKey = ""; state.tactileSeriesPromise = null; state.tactileAppliedKey = ""; state.tactilePendingFrame = null; state.tactileGeneration += 1; loadAnnotateVideo(); seekFrame(0); });''',
    "camera tactile queue reset",
)


# ---------------------------------------------------------------------------
# Detailed tactile viewer: same one-batch-at-a-time rule.
# ---------------------------------------------------------------------------
app = ROOT / "webui/static/tactile/app.js"
replace_once(
    app,
    '''    currentFrame: 0,\n    lastSyncKey: "",\n    videoFrameCallbackId: 0,''',
    '''    currentFrame: 0,\n    lastSyncKey: "",\n    imageLoadActive: false,\n    pendingRender: null,\n    renderSerial: 0,\n    videoFrameCallbackId: 0,''',
    "viewer tactile queue state",
)
replace_once(
    app,
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    grid.innerHTML = "";''',
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    state.pendingRender = null;\n    state.renderSerial += 1;\n    grid.innerHTML = "";''',
    "viewer queue reset",
)
old_render = '''  function renderSynchronizedFrame(record, camera, match, frame, force) {\n    if (!match || !match.row) {\n      statusNode.textContent = "No synchronized tactile frame.";\n      grid.innerHTML = "";\n      return;\n    }\n    var syncRow = match.row;\n    var kind = String(byId("kindSelect").value || "deform");\n    var syncKey = record.id + "|" + camera + "|" + syncRow.sync_row + "|" + kind;\n    if (!force && syncKey === state.lastSyncKey) {\n      updateCurvePlayhead(frame);\n      return;\n    }\n    state.lastSyncKey = syncKey;\n    grid.style.setProperty("--tactile-cell-aspect", kind === "raw" ? "4 / 3" : "1 / 1");\n    var completeText = syncRow.complete === false ? "incomplete" : "complete";\n    statusNode.textContent = camera + " frame " + frame\n      + " → sync frame " + syncRow.frame\n      + " · row " + syncRow.sync_row + " · " + completeText;\n    grid.innerHTML = FINGERS.map(function (finger) {\n      return renderFinger(record, finger, currentFingerData(syncRow, finger), kind);\n    }).join("");\n    updateCurvePlayhead(frame);\n  }\n\n  async function refreshTactile(frame, force) {\n    var record = selectedRecord();\n    if (!record) return;\n    updateReadout(frame);\n    var series = await ensureSeries(record, state.camera);\n    if (!series) return;\n    renderSynchronizedFrame(record, state.camera, nearestSyncFrame(state.currentFrame), state.currentFrame, Boolean(force));\n  }\n'''
new_render = '''  function preloadTactileImage(url) {\n    return new Promise(function (resolve, reject) {\n      var image = new Image();\n      image.onload = resolve;\n      image.onerror = function () { reject(new Error("tactile image failed")); };\n      image.src = url;\n    });\n  }\n\n  async function applySynchronizedFrame(record, camera, match, frame, force, serial) {\n    if (!match || !match.row) {\n      statusNode.textContent = "No synchronized tactile frame.";\n      return;\n    }\n    var syncRow = match.row;\n    var kind = String(byId("kindSelect").value || "deform");\n    var syncKey = record.id + "|" + camera + "|" + syncRow.sync_row + "|" + kind;\n    if (!force && syncKey === state.lastSyncKey) {\n      updateCurvePlayhead(frame);\n      return;\n    }\n    var fingerData = {};\n    var loads = [];\n    FINGERS.forEach(function (finger) {\n      var data = currentFingerData(syncRow, finger);\n      fingerData[finger] = data;\n      var kinds = data && Array.isArray(data.image_kinds) ? data.image_kinds : [];\n      if (data && data.event_id != null && kinds.indexOf(kind) >= 0) {\n        loads.push(preloadTactileImage(tactileImageUrl(record, finger, data.event_id, kind)));\n      }\n    });\n    var completeText = syncRow.complete === false ? "incomplete" : "complete";\n    statusNode.textContent = camera + " frame " + frame + " → sync frame " + syncRow.frame + " · loading " + kind;\n    try {\n      await Promise.all(loads);\n    } catch (_error) {\n      if (serial === state.renderSerial) statusNode.textContent = "Tactile image load failed; keeping previous frame";\n      return;\n    }\n    if (serial !== state.renderSerial || record.id !== state.selectedId || camera !== state.camera) return;\n    grid.style.setProperty("--tactile-cell-aspect", kind === "raw" ? "4 / 3" : "1 / 1");\n    grid.innerHTML = FINGERS.map(function (finger) {\n      return renderFinger(record, finger, fingerData[finger], kind);\n    }).join("");\n    state.lastSyncKey = syncKey;\n    statusNode.textContent = camera + " frame " + frame\n      + " → sync frame " + syncRow.frame\n      + " · row " + syncRow.sync_row + " · " + completeText;\n    updateCurvePlayhead(frame);\n  }\n\n  async function drainSynchronizedFrameQueue() {\n    if (state.imageLoadActive) return;\n    state.imageLoadActive = true;\n    var serial = state.renderSerial;\n    try {\n      while (state.pendingRender && serial === state.renderSerial) {\n        var pending = state.pendingRender;\n        state.pendingRender = null;\n        await applySynchronizedFrame(pending.record, pending.camera, pending.match, pending.frame, pending.force, serial);\n      }\n    } finally {\n      state.imageLoadActive = false;\n      if (state.pendingRender) Promise.resolve().then(drainSynchronizedFrameQueue);\n    }\n  }\n\n  function requestSynchronizedFrame(record, camera, match, frame, force) {\n    state.pendingRender = { record: record, camera: camera, match: match, frame: frame, force: force };\n    updateCurvePlayhead(frame);\n    if (!state.imageLoadActive) drainSynchronizedFrameQueue();\n  }\n\n  async function refreshTactile(frame, force) {\n    var record = selectedRecord();\n    if (!record) return;\n    updateReadout(frame);\n    var series = await ensureSeries(record, state.camera);\n    if (!series) return;\n    requestSynchronizedFrame(record, state.camera, nearestSyncFrame(state.currentFrame), state.currentFrame, Boolean(force));\n  }\n'''
replace_once(app, old_render, new_render, "viewer tactile backpressure")
replace_once(
    app,
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    setVideoSource(record, true);''',
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    state.pendingRender = null;\n    state.renderSerial += 1;\n    setVideoSource(record, true);''',
    "viewer camera queue reset",
)
replace_once(
    app,
    '''  byId("kindSelect").addEventListener("change", function () {\n    state.lastSyncKey = "";\n    refreshTactile(state.currentFrame, true);\n  });''',
    '''  byId("kindSelect").addEventListener("change", function () {\n    state.lastSyncKey = "";\n    state.pendingRender = null;\n    state.renderSerial += 1;\n    refreshTactile(state.currentFrame, true);\n  });''',
    "viewer kind queue reset",
)
