"use strict";

var LABEL_KEYS = [1, 2, 3, 4];
var TACTILE_FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
var state = {
  view: "annotate",
  rollouts: [],
  filtered: [],
  selectedId: null,
  currentFrame: 0,
  sidebarCollapsed: false,
  annotateCamera: "",
  tactileF6Lookup: {},
  events: [],
  activeEvent: null,
  dirty: false,
  settings: null,
  labels: [],
  labelDraft: null,
  annotationSource: null,
  annotationTarget: null,
  videoGeneration: 0,
  tactileGeneration: 0,
  tactileSeries: null,
  tactileSeriesKey: "",
  tactileSeriesPromise: null,
  tactileAppliedKey: "",
  tactileLoadActive: false,
  tactilePendingFrame: null,
  results: {
    sources: [], selectedSource: "", selectedId: "", currentFrame: 0,
    points: [], videoGeneration: 0, annotationLabels: [], visibleAnnotationLabels: []
  },
  telemetry: [],
  telemetryTimer: null,
  videoDiagnostics: {}
};

function byId(id) { return document.getElementById(id); }
function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function taskLabel(record) {
  return String(record.task_description || record.instruction || record.task_key || ("Task " + (record.task_id == null ? "" : record.task_id)) || "Tactile task");
}
function formatTime(seconds) {
  var value = Math.max(0, Number(seconds) || 0);
  var minutes = Math.floor(value / 60);
  var secs = Math.floor(value % 60);
  var millis = Math.floor((value - Math.floor(value)) * 1000);
  return String(minutes).padStart(2, "0") + ":" + String(secs).padStart(2, "0") + "." + String(millis).padStart(3, "0");
}
function formatPercent(value) { return (100 * (Number(value) || 0)).toFixed(1) + "%"; }
function formatDate(value) {
  if (!value) return "";
  var date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
function isTypingTarget(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].indexOf(target.tagName) >= 0;
}
function labelMeta(id) { return state.labels.find(function (row) { return row.id === Number(id); }) || null; }
function labelName(id) { var meta = labelMeta(id); return meta ? meta.name : "Unknown " + id; }
function eligibleLabel(label, record) { return Boolean(label && label.active && (label.scope !== "failure" || record && record.ground_truth_outcome === "failure")); }
function displayedLabel(id) { return id + " · " + labelName(id); }
function badge(label, cssClass) {
  return '<span class="badge ' + escapeHtml(cssClass || "") + '">' + escapeHtml(label) + "</span>";
}
function connectionSnapshot() {
  var connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  if (!connection) return null;
  return {
    effective_type: connection.effectiveType || null,
    downlink_mbps: connection.downlink == null ? null : connection.downlink,
    rtt_ms: connection.rtt == null ? null : connection.rtt,
    save_data: Boolean(connection.saveData)
  };
}

function queueTelemetry(payload) {
  state.telemetry.push(Object.assign({
    ts_client: new Date().toISOString(),
    page: state.view,
    connection: connectionSnapshot()
  }, payload || {}));
  if (state.telemetry.length > 100) state.telemetry.splice(0, state.telemetry.length - 100);
  if (!state.telemetryTimer) {
    state.telemetryTimer = window.setTimeout(flushTelemetry, 900);
  }
}
async function flushTelemetry() {
  state.telemetryTimer = null;
  if (!state.telemetry.length) return;
  var events = state.telemetry.splice(0, 50);
  try {
    await fetch("/api/telemetry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: events }),
      cache: "no-store",
      keepalive: true
    });
  } catch (_error) {
    /* Telemetry must never make the UI unusable. */
  }
  if (state.telemetry.length && !state.telemetryTimer) state.telemetryTimer = window.setTimeout(flushTelemetry, 900);
}

async function jsonRequest(url, options) {
  var started = performance.now();
  var response;
  try {
    response = await fetch(url, Object.assign({ cache: "no-store" }, options || {}));
  } catch (error) {
    queueTelemetry({ event: "network_error", level: "error", url: url, duration_ms: performance.now() - started, message: String(error && error.message || error) });
    throw error;
  }
  var text = await response.text();
  var payload = {};
  if (text) {
    try { payload = JSON.parse(text); }
    catch (_error) { payload = { error: "Invalid JSON response" }; }
  }
  queueTelemetry({ event: "api_request", level: response.ok ? "info" : "error", url: url, status: response.status, duration_ms: performance.now() - started });
  if (!response.ok) throw new Error(payload.error || ("Request failed: " + response.status));
  return payload;
}

function applyTheme(theme) {
  document.body.classList.toggle("theme-dark", theme === "dark");
  document.documentElement.dataset.appearance = theme === "dark" ? "dark" : "light";
}
function setGlobalStatus(ok, text) {
  byId("globalStatus").classList.toggle("online", ok === true);
  byId("globalStatus").classList.toggle("error", ok === false);
  byId("globalStatusText").textContent = text;
}
function switchView(view, updateHash) {
  if (!byId("view-" + view)) return;
  if (state.view === "annotate" && view !== "annotate" && state.dirty) {
    if (!window.confirm("Discard unsaved interval changes?")) return;
    state.dirty = false;
  }
  state.view = view;
  document.body.dataset.view = view;
  document.querySelectorAll(".view").forEach(function (node) { node.classList.toggle("active-view", node.id === "view-" + view); });
  document.querySelectorAll(".nav-item").forEach(function (button) { button.classList.toggle("active", button.dataset.view === view); });
  if (updateHash !== false) history.replaceState(null, "", "#" + view);
  if (view === "analysis") loadAnalysis();
  if (view === "runs") loadRuns();
  if (view === "settings") { loadSettings(); loadDiagnostics(); }
  if (view === "results") loadResultsSources(false);
}

function selectedRollout() {
  return state.rollouts.find(function (record) { return record.id === state.selectedId; }) || null;
}
function resultRollout() {
  return state.rollouts.find(function (record) { return record.id === state.results.selectedId; }) || null;
}
function updateHeader() {
  var complete = state.rollouts.filter(function (record) { return record.annotation_status === "complete"; }).length;
  byId("datasetStatus").textContent = state.rollouts.length + " rollouts";
  byId("annotationStatus").textContent = complete + " / " + state.rollouts.length + " annotated";
}
function populateTaskFilter() {
  var select = byId("annotateTaskFilter");
  var current = select.value || "all";
  var tasks = Array.from(new Set(state.rollouts.map(taskLabel))).sort();
  select.innerHTML = '<option value="all">All tasks</option>' + tasks.map(function (task) {
    return '<option value="' + escapeHtml(task) + '">' + escapeHtml(task) + "</option>";
  }).join("");
  select.value = tasks.indexOf(current) >= 0 ? current : "all";
}
function filterRollouts() {
  var query = byId("annotateSearch").value.trim().toLowerCase();
  var task = byId("annotateTaskFilter").value;
  var review = byId("annotateReviewFilter").value;
  state.filtered = state.rollouts.filter(function (record) {
    var haystack = [record.id, taskLabel(record), record.task_key, record.task_id, record.episode_index].join(" ").toLowerCase();
    return (!query || haystack.indexOf(query) >= 0)
      && (task === "all" || taskLabel(record) === task)
      && (review === "all" || record.annotation_status === review);
  });
  byId("annotateVisibleCount").textContent = state.filtered.length;
  renderRolloutList();
}
function renderRolloutList() {
  var container = byId("annotateRolloutList");
  if (!state.filtered.length) {
    container.innerHTML = '<div class="muted" style="padding:18px 4px;font-size:.65rem">No matching rollouts.</div>';
    return;
  }
  container.innerHTML = state.filtered.map(function (record) {
    var events = record.annotation_events || [];
    return '<button type="button" class="rollout-card' + (record.id === state.selectedId ? " active" : "") + '" data-rollout-id="' + escapeHtml(record.id) + '">'
      + '<div class="badge-row">' + badge(record.annotation_status || "unreviewed", record.annotation_status || "")
      + (events.length ? badge(events.length + " intervals", "") : "") + '</div>'
      + '<div class="rollout-title">' + escapeHtml(taskLabel(record)) + '</div>'
      + '<div class="rollout-meta"><span>' + escapeHtml(record.id) + '</span><span>' + escapeHtml(record.total_frames || "?") + 'f</span></div></button>';
  }).join("");
  container.querySelectorAll("[data-rollout-id]").forEach(function (button) {
    button.addEventListener("click", function () { maybeSelectRollout(button.dataset.rolloutId); });
  });
}
function maybeSelectRollout(id) {
  if (state.dirty && id !== state.selectedId && !window.confirm("Discard unsaved interval changes?")) return;
  selectRollout(id);
}
var CAMERA_ORDER = ["cam_high", "cam_wrist", "cam_left_wrist", "cam_right_wrist"];
var CAMERA_LABELS = { cam_high: "High", cam_wrist: "Wrist", cam_left_wrist: "Left wrist", cam_right_wrist: "Right wrist" };
function cameraKeys(record) {
  var paths = record && record.camera_video_paths;
  if (!paths || typeof paths !== "object") return [];
  return Object.keys(paths).filter(function (key) { return typeof paths[key] === "string" && paths[key].trim(); }).sort(function (a, b) {
    var i = CAMERA_ORDER.indexOf(a), j = CAMERA_ORDER.indexOf(b);
    return (i < 0 ? 99 : i) - (j < 0 ? 99 : j) || a.localeCompare(b);
  });
}
function chooseCamera(record) {
  var keys = cameraKeys(record);
  if (!keys.length) return "";
  var preferred = state.annotateCamera;
  if (!preferred) { try { preferred = sessionStorage.getItem("tactile.annotate.camera") || ""; } catch (_error) {} }
  if (preferred && keys.indexOf(preferred) >= 0) return preferred;
  preferred = state.settings && state.settings.default_camera;
  if (preferred && keys.indexOf(preferred) >= 0) return preferred;
  if (record.observation_key && keys.indexOf(record.observation_key) >= 0) return record.observation_key;
  if (keys.indexOf("cam_high") >= 0) return "cam_high";
  return keys[0];
}
function cameraLabel(key) {
  return CAMERA_LABELS[key] || String(key).replace(/^cam_/, "").replace(/_/g, " ").replace(/\b\w/g, function (letter) { return letter.toUpperCase(); });
}
function renderAnnotateCameras(record) {
  var node = byId("annotateCamera"), keys = cameraKeys(record);
  node.classList.toggle("hidden", keys.length < 2);
  node.innerHTML = keys.length < 2 ? "" : '<span class="camera-view-label">View</span><div class="camera-view-buttons">'
    + keys.map(function (key) {
      var active = key === state.annotateCamera;
      return '<button type="button" class="camera-view-button' + (active ? ' is-active' : '') + '" data-camera="' + escapeHtml(key) + '" aria-pressed="' + active + '">' + escapeHtml(cameraLabel(key)) + '</button>';
    }).join("") + '</div>';
}
function setRolloutSidebarCollapsed(collapsed) {
  state.sidebarCollapsed = Boolean(collapsed);
  byId("annotateSidebar").parentElement.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  var button = byId("toggleRolloutSidebar");
  button.textContent = state.sidebarCollapsed ? "›" : "‹";
  button.title = state.sidebarCollapsed ? "Expand rollout list" : "Collapse rollout list";
  button.setAttribute("aria-expanded", String(!state.sidebarCollapsed));
  try { sessionStorage.setItem("tactile.annotate.sidebarCollapsed", state.sidebarCollapsed ? "1" : "0"); } catch (_error) {}
}
function switchAnnotateCamera(camera) {
  var record = selectedRollout();
  if (!record || cameraKeys(record).indexOf(camera) < 0 || state.annotateCamera === camera) return;
  var frame = state.currentFrame, video = byId("annotateVideo"), wasPlaying = !video.paused && !video.ended;
  state.annotateCamera = camera;
  try { sessionStorage.setItem("tactile.annotate.camera", camera); } catch (_error) {}
  state.tactileGeneration += 1;
  state.tactileSeries = null; state.tactileSeriesKey = ""; state.tactileSeriesPromise = null;
  state.tactileF6Lookup = {}; state.tactileAppliedKey = ""; state.tactilePendingFrame = null;
  byId("annotateF6Curve").innerHTML = "Loading F6 history…";
  renderAnnotateCameras(record);
  loadAnnotateVideo(frame, wasPlaying);
  updateAnnotateTactile(frame);
}

function timelinePercent(frame, total) {
  return total <= 1 ? 0 : Math.max(0, Math.min(100, Number(frame) * 100 / (total - 1)));
}
function renderPrecisionTimeline(containerId, events, totalFrames, currentFrame, activeIndex, onSelect, onSeek, labelKeys) {
  var container = byId(containerId);
  if (!container) return;
  var total = Math.max(1, Number(totalFrames) || 1);
  var ticks = [0, .25, .5, .75, 1];
  var html = '<div class="timeline-track-area">';
  ticks.forEach(function (ratio) {
    html += '<i class="timeline-gridline" style="left:' + (ratio * 100) + '%"></i>';
    html += '<span class="timeline-axis-label" style="left:' + (ratio * 100) + '%">' + Math.round((total - 1) * ratio) + '</span>';
  });
  var lanes = Array.isArray(labelKeys) ? labelKeys : Array.from(new Set(LABEL_KEYS.concat((events || []).map(function (event) { return Number(event.event_key); })))).sort(function (a, b) { return a - b; });
  lanes.forEach(function (key) {
    var meta = labelMeta(key);
    html += '<div class="timeline-lane" data-label-key="' + key + '" title="' + escapeHtml(meta ? meta.name + ": " + meta.description : "Unregistered label " + key) + '"><span class="timeline-lane-label">' + key + '</span>';
    (events || []).forEach(function (event, index) {
      if (Number(event.event_key) !== key) return;
      var left = timelinePercent(event.start_frame, total);
      var right = timelinePercent(event.end_frame, total);
      var width = Math.max(.45, right - left + 100 / total);
      html += '<div class="timeline-segment label-' + key + (index === activeIndex ? ' active' : '') + '" data-event-index="' + index + '" style="left:' + left + '%;width:' + Math.min(width, 100 - left) + '%;' + (meta ? 'background:' + meta.color + ';' : '') + '"><span>' + key + '</span></div>';
    });
    html += '</div>';
  });
  html += '<div class="timeline-playhead" style="left:' + timelinePercent(currentFrame, total) + '%"></div></div>';
  container.innerHTML = html;
  var track = container.querySelector(".timeline-track-area");
  if (track && onSeek) {
    track.addEventListener("click", function (event) {
      if (event.target.closest(".timeline-segment")) return;
      var box = track.getBoundingClientRect();
      var ratio = Math.max(0, Math.min(1, (event.clientX - box.left) / Math.max(1, box.width)));
      onSeek(Math.round(ratio * (total - 1)));
    });
  }
  container.querySelectorAll(".timeline-segment").forEach(function (segment) {
    segment.addEventListener("click", function (event) {
      event.stopPropagation();
      var index = Number(segment.dataset.eventIndex);
      if (onSelect) onSelect(index);
      var item = events[index];
      if (item && onSeek) onSeek(item.start_frame);
    });
  });
}
function updateTimelinePlayhead(containerId, frame, totalFrames) {
  var node = byId(containerId);
  if (!node) return;
  var playhead = node.querySelector(".timeline-playhead");
  if (playhead) playhead.style.left = timelinePercent(frame, Math.max(1, Number(totalFrames) || 1)) + "%";
}

function selectRollout(id) {
  var record = state.rollouts.find(function (item) { return item.id === id; });
  if (!record) return;
  state.selectedId = id;
  state.currentFrame = 0;
  state.events = (record.annotation_events || []).map(function (event) { return Object.assign({}, event); });
  state.activeEvent = state.events.length ? 0 : null;
  state.dirty = false;
  state.tactileSeries = null;
  state.tactileSeriesKey = "";
  state.tactileSeriesPromise = null;
  state.tactileF6Lookup = {};
  state.tactileAppliedKey = "";
  state.tactilePendingFrame = null;
  state.tactileGeneration += 1;
  byId("annotateEmpty").classList.add("hidden");
  byId("annotateContent").classList.remove("hidden");
  renderRolloutList();
  state.annotateCamera = chooseCamera(record);
  renderAnnotateCameras(record);
  byId("annotateTaskTitle").textContent = taskLabel(record);
  byId("annotateRecordMeta").textContent = record.id + " · " + (record.total_frames || "?") + " frames · " + (record.fps || "?") + " fps";
  var slider = byId("annotateFrameSlider");
  slider.max = Math.max(0, Number(record.total_frames || 1) - 1);
  slider.value = "0";
  byId("annotateF6Curve").innerHTML = "Loading F6 history…";
  loadAnnotateVideo(0, false);
  renderIntervals();
  renderAnnotateTimeline();
  updateFrameUi(0, true);
  updateNavigationButtons();
  setAnnotationMessage("", "");
}
function renderAnnotateTimeline() {
  var record = selectedRollout();
  if (!record) return;
  renderPrecisionTimeline("annotateTimeline", state.events, record.total_frames, state.currentFrame, state.activeEvent,
    function (index) { state.activeEvent = index; renderIntervals(); renderAnnotateTimeline(); }, seekFrame);
}
function loadAnnotateVideo(preserveFrame, resumePlayback) {
  var record = selectedRollout();
  if (!record) return;
  var camera = state.annotateCamera || chooseCamera(record);
  var video = byId("annotateVideo");
  var frame = clampFrame(preserveFrame == null ? state.currentFrame : preserveFrame);
  state.videoGeneration += 1;
  var generation = state.videoGeneration;
  video.pause();
  video.addEventListener("loadedmetadata", function () {
    if (generation !== state.videoGeneration || record.id !== state.selectedId) return;
    try { video.currentTime = frame / (Number(record.fps) || 30); } catch (_error) {}
    updateFrameUi(frame, true);
    if (resumePlayback) video.play().catch(function () {});
  }, { once: true });
  video.src = "/api/videos/" + encodeURIComponent(record.id) + "?camera=" + encodeURIComponent(camera);
  video.load();
  byId("annotatePlay").textContent = "Play";
  if (typeof video.requestVideoFrameCallback === "function") {
    var watch = function (_now, metadata) {
      if (generation !== state.videoGeneration) return;
      if (!video.paused && !video.ended) updateFrameUi(Math.round((metadata.mediaTime || video.currentTime) * (Number(record.fps) || 30)), false);
      video.requestVideoFrameCallback(watch);
    };
    video.requestVideoFrameCallback(watch);
  }
}
function clampFrame(frame) {
  var record = selectedRollout();
  var max = Math.max(0, Number(record && record.total_frames || 1) - 1);
  return Math.max(0, Math.min(max, Math.round(Number(frame) || 0)));
}
function seekFrame(frame) {
  var record = selectedRollout();
  if (!record) return;
  var value = clampFrame(frame);
  state.currentFrame = value;
  byId("annotateVideo").currentTime = value / (Number(record.fps) || 30);
  updateFrameUi(value, true);
}
function updateFrameUi(frame, forceTactile) {
  var record = selectedRollout();
  if (!record) return;
  var value = clampFrame(frame);
  var changed = value !== state.currentFrame;
  state.currentFrame = value;
  byId("annotateFrameSlider").value = String(value);
  byId("annotateFrameText").textContent = "Frame " + value + " / " + Math.max(0, Number(record.total_frames || 1) - 1);
  byId("annotateTimeText").textContent = formatTime(value / (Number(record.fps) || 30));
  updateTimelinePlayhead("annotateTimeline", value, record.total_frames);
  updateAnnotateF6Playhead(value);
  if (changed || forceTactile) updateAnnotateTactile(value);
}
function annotateTactileImageUrl(record, finger, eventId, kind) {
  return "/api/tactile/" + encodeURIComponent(record.id) + "/image?finger=" + encodeURIComponent(finger)
    + "&event_id=" + encodeURIComponent(eventId) + "&kind=" + encodeURIComponent(kind);
}
function nearestAnnotateTactileSync(series, frame) {
  var rows = series && Array.isArray(series.sync_frames) ? series.sync_frames : [];
  if (!rows.length) return null;
  var low = 0, high = rows.length - 1;
  while (low <= high) {
    var mid = (low + high) >> 1;
    var value = Number(rows[mid].frame);
    if (value < frame) low = mid + 1; else if (value > frame) high = mid - 1; else return rows[mid];
  }
  if (low <= 0) return rows[0];
  if (low >= rows.length) return rows[rows.length - 1];
  return Math.abs(frame - Number(rows[low - 1].frame)) <= Math.abs(Number(rows[low].frame) - frame) ? rows[low - 1] : rows[low];
}
function ensureAnnotateTactileSeries(record, camera) {
  var key = record.id + "|" + camera;
  if (state.tactileSeriesKey === key && state.tactileSeries) return Promise.resolve(state.tactileSeries);
  if (state.tactileSeriesKey === key && state.tactileSeriesPromise) return state.tactileSeriesPromise;
  state.tactileSeriesKey = key;
  state.tactileSeries = null;
  state.tactileAppliedKey = "";
  state.tactileSeriesPromise = jsonRequest("/api/tactile/" + encodeURIComponent(record.id) + "/series?camera=" + encodeURIComponent(camera))
    .then(function (payload) {
      if (state.tactileSeriesKey !== key) return null;
      state.tactileSeries = payload.tactile || null;
      state.tactileF6Lookup = {};
      TACTILE_FINGERS.forEach(function (finger) {
        var events = state.tactileSeries && state.tactileSeries.fingers && state.tactileSeries.fingers[finger] || [];
        state.tactileF6Lookup[finger] = new Map(events.map(function (row) { return [String(row.event_id), row]; }));
      });
      renderAnnotateF6Curve();
      return state.tactileSeries;
    }).catch(function (error) {
      if (state.tactileSeriesKey === key) byId("annotateTactileStatus").textContent = "Tactile timeline unavailable: " + String(error.message || error);
      return null;
    }).finally(function () {
      if (state.tactileSeriesKey === key) state.tactileSeriesPromise = null;
    });
  return state.tactileSeriesPromise;
}
async function loadAnnotateTactileFrame(frame, generation) {
  var record = selectedRollout();
  if (!record) return;
  var camera = byId("annotateCamera").value || chooseCamera(record);
  var kind = state.settings && state.settings.default_tactile_kind || "deform";
  var series = await ensureAnnotateTactileSeries(record, camera);
  if (generation !== state.tactileGeneration || !series) return;
  var sync = nearestAnnotateTactileSync(series, frame);
  if (!sync) { byId("annotateTactileStatus").textContent = "No synchronized tactile at this frame"; return; }
  var fingerRows = sync.fingers || {};
  var key = record.id + "|" + camera + "|" + sync.frame + "|" + kind + "|"
    + TACTILE_FINGERS.map(function (finger) { return fingerRows[finger] && fingerRows[finger].event_id != null ? fingerRows[finger].event_id : "-"; }).join(",");
  if (key === state.tactileAppliedKey) return;
  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · loading " + kind;
  var cells = [];
  var loads = [];
  var started = performance.now();
  TACTILE_FINGERS.forEach(function (finger) {
    var info = fingerRows[finger] || {};
    if (info.event_id == null) { cells.push({ finger: finger, missing: true }); return; }
    var url = annotateTactileImageUrl(record, finger, info.event_id, kind);
    cells.push({ finger: finger, url: url });
    loads.push(new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = resolve;
      image.onerror = function () { reject(new Error(finger + " tactile image failed")); };
      image.src = url;
    }));
  });
  try {
    await Promise.all(loads);
  } catch (error) {
    if (generation !== state.tactileGeneration) return;
    byId("annotateTactileStatus").textContent = "Tactile image load failed; keeping previous frame";
    queueTelemetry({ event: "tactile_image_error", level: "error", rollout_id: record.id, camera: camera, frame: frame, matched_frame: sync.frame, kind: kind, message: String(error.message || error) });
    return;
  }
  if (generation !== state.tactileGeneration || record.id !== state.selectedId) return;
  byId("annotateTactileGrid").innerHTML = cells.map(function (cell) {
    return '<div class="annotate-tactile-finger"><strong>' + escapeHtml(cell.finger) + '</strong>'
      + (cell.missing ? '<div class="tactile-missing">No sample</div>' : '<img src="' + escapeHtml(cell.url) + '" alt="' + escapeHtml(cell.finger + " " + kind) + '">') + '</div>';
  }).join("");
  state.tactileAppliedKey = key;
  var duration = performance.now() - started;
  if (duration >= 250) queueTelemetry({ event: "tactile_batch_slow", level: duration >= 1000 ? "error" : "info", rollout_id: record.id, camera: camera, frame: frame, matched_frame: sync.frame, kind: kind, duration_ms: duration });
  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · " + kind;
}
async function drainAnnotateTactileQueue() {
  if (state.tactileLoadActive) return;
  state.tactileLoadActive = true;
  var generation = state.tactileGeneration;
  try {
    while (state.tactilePendingFrame != null && generation === state.tactileGeneration) {
      var frame = state.tactilePendingFrame;
      state.tactilePendingFrame = null;
      await loadAnnotateTactileFrame(frame, generation);
    }
  } finally {
    state.tactileLoadActive = false;
    if (state.tactilePendingFrame != null) Promise.resolve().then(drainAnnotateTactileQueue);
  }
}
function updateAnnotateTactile(frame) {
  state.tactilePendingFrame = clampFrame(frame);
  if (!state.tactileLoadActive) drainAnnotateTactileQueue();
}
function togglePlay() {
  var video = byId("annotateVideo");
  if (video.paused || video.ended) video.play().catch(function () {}); else video.pause();
}
function eventOptions(selected) {
  var record = selectedRollout();
  var options = state.labels.filter(function (label) {
    return eligibleLabel(label, record) || label.id === Number(selected);
  }).map(function (label) {
    return '<option value="' + label.id + '"' + (label.id === Number(selected) ? ' selected' : '') + '>' + escapeHtml(displayedLabel(label.id)) + (label.active ? '' : ' (inactive)') + '</option>';
  });
  if (!labelMeta(selected)) options.unshift('<option selected value="' + escapeHtml(selected) + '">Unregistered label ' + escapeHtml(selected) + '</option>');
  return options.join('');
}
function renderAnnotationLabelGuide() {
  var node = byId("annotationLabelGuide");
  if (!node) return;
  var record = selectedRollout();
  node.innerHTML = state.labels.filter(function (label) {
    return label.active || state.events.some(function (event) { return Number(event.event_key) === label.id; });
  }).map(function (label) {
    var eligible = eligibleLabel(label, record);
    return '<div class="annotation-label-definition' + (eligible ? '' : ' ineligible') + '" title="' + escapeHtml(label.description) + '"><i class="label-mark" style="background:' + label.color + '"></i><strong>' + label.id + ' · ' + escapeHtml(label.name) + '</strong><span class="label-explanation">' + escapeHtml(label.description) + (label.scope === 'failure' ? ' · failure only' : '') + (label.active ? '' : ' · inactive') + '</span></div>';
  }).join('');
}
function renderActiveLabelHelp() {
  var node = byId("activeLabelHelp"); if (!node) return;
  var event = state.activeEvent == null ? null : state.events[state.activeEvent];
  if (!event) { node.textContent = 'Select a label to see its definition.'; node.className = 'active-label-help'; return; }
  var label = labelMeta(event.event_key);
  node.textContent = label ? displayedLabel(label.id) + ': ' + label.description + (label.active ? '' : ' (inactive)') + (label.scope === 'failure' ? ' · failure rollout only' : '') : 'Unregistered label ' + event.event_key + ': preserved unchanged. Register this ID or explicitly reassign before saving.';
  node.className = 'active-label-help' + (!label || label.scope === 'failure' && !eligibleLabel(label, selectedRollout()) ? ' warning' : '');
}
function normalizeLocalEvents() {
  state.events.forEach(function (event, index) {
    event.event_index = index;
    // Unknown historical event IDs must never be silently rewritten to label 1.
    event.event_key = Number(event.event_key);
    event.start_frame = clampFrame(event.start_frame);
    event.end_frame = clampFrame(event.end_frame);
    if (event.end_frame < event.start_frame) event.end_frame = event.start_frame;
  });
}
function renderIntervals() {
  normalizeLocalEvents();
  var container = byId("intervalList");
  if (!state.events.length) {
    container.innerHTML = '<div class="muted" style="padding:18px 3px;font-size:.64rem">No intervals yet. Add one at the current frame.</div>';
    renderAnnotateTimeline(); renderAnnotationLabelGuide(); renderActiveLabelHelp();
    return;
  }
  container.innerHTML = state.events.map(function (event, index) {
    return '<div class="interval-row' + (index === state.activeEvent ? " active" : "") + '" data-event-index="' + index + '">'
      + '<div class="interval-index">' + (index + 1) + '</div>'
      + '<label class="interval-field"><span>Label</span><select class="event-type">' + eventOptions(event.event_key) + '</select></label>'
      + '<label class="interval-field"><span>Start</span><input class="event-start" type="number" min="0" value="' + event.start_frame + '"><div class="interval-frame-tools"><button type="button" class="use-start">Current Q</button></div></label>'
      + '<label class="interval-field"><span>End</span><input class="event-end" type="number" min="0" value="' + event.end_frame + '"><div class="interval-frame-tools"><button type="button" class="use-end">Current W</button></div></label>'
      + '<div class="interval-actions"><button type="button" class="jump-start">Start</button><button type="button" class="jump-end">End</button><button type="button" class="delete-event">Delete</button></div></div>';
  }).join("");
  container.querySelectorAll(".interval-row").forEach(function (row) {
    var index = Number(row.dataset.eventIndex);
    row.addEventListener("click", function (event) { if (event.target.closest("input,select,button")) return; state.activeEvent = index; renderIntervals(); renderAnnotateTimeline(); });
    row.addEventListener("focusin", function () { state.activeEvent = index; container.querySelectorAll(".interval-row").forEach(function (item) { item.classList.toggle("active", Number(item.dataset.eventIndex) === index); }); renderAnnotateTimeline(); });
    row.querySelector(".event-type").addEventListener("change", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].event_key = Number(event.target.value); markDirty(); renderIntervals(); renderAnnotateTimeline(); });
    row.querySelector(".event-start").addEventListener("change", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].start_frame = clampFrame(event.target.value); if (state.events[index].end_frame < state.events[index].start_frame) state.events[index].end_frame = state.events[index].start_frame; markDirty(); renderIntervals(); renderAnnotateTimeline(); });
    row.querySelector(".event-end").addEventListener("change", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].end_frame = Math.max(state.events[index].start_frame, clampFrame(event.target.value)); markDirty(); renderIntervals(); renderAnnotateTimeline(); });
    row.querySelector(".use-start").addEventListener("click", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].start_frame = state.currentFrame; if (state.events[index].end_frame < state.currentFrame) state.events[index].end_frame = state.currentFrame; markDirty(); renderIntervals(); renderAnnotateTimeline(); });
    row.querySelector(".use-end").addEventListener("click", function (event) { event.stopPropagation(); state.activeEvent = index; state.events[index].end_frame = Math.max(state.events[index].start_frame, state.currentFrame); markDirty(); renderIntervals(); renderAnnotateTimeline(); });
    row.querySelector(".jump-start").addEventListener("click", function (event) { event.stopPropagation(); seekFrame(state.events[index].start_frame); });
    row.querySelector(".jump-end").addEventListener("click", function (event) { event.stopPropagation(); seekFrame(state.events[index].end_frame); });
    row.querySelector(".delete-event").addEventListener("click", function (event) { event.stopPropagation(); state.events.splice(index, 1); state.activeEvent = state.events.length ? Math.min(index, state.events.length - 1) : null; markDirty(); renderIntervals(); renderAnnotateTimeline(); });
  });
  renderAnnotationLabelGuide(); renderActiveLabelHelp();
}
function addInterval(key) {
  var record = selectedRollout();
  var label = labelMeta(key);
  if (!eligibleLabel(label, record)) { setAnnotationMessage('Label ' + key + ' is unavailable for this rollout.', 'error'); return; }
  state.events.push({ event_key: label.id, start_frame: state.currentFrame, end_frame: state.currentFrame, notes: "" });
  state.activeEvent = state.events.length - 1; markDirty(); renderIntervals(); renderAnnotateTimeline();
}
function setActiveEventType(key) {
  if (!eligibleLabel(labelMeta(key), selectedRollout())) { setAnnotationMessage('Label ' + key + ' is unavailable for this rollout.', 'error'); return; }
  if (state.activeEvent == null || !state.events[state.activeEvent]) addInterval(key);
  else { state.events[state.activeEvent].event_key = Number(key); markDirty(); renderIntervals(); renderAnnotateTimeline(); }
}
function setActiveBoundary(which) {
  if (state.activeEvent == null || !state.events[state.activeEvent]) return;
  var event = state.events[state.activeEvent];
  if (which === "start") { event.start_frame = state.currentFrame; if (event.end_frame < event.start_frame) event.end_frame = event.start_frame; }
  else event.end_frame = Math.max(event.start_frame, state.currentFrame);
  markDirty(); renderIntervals(); renderAnnotateTimeline();
}
function markDirty() { state.dirty = true; setAnnotationMessage("Unsaved changes", ""); }
function setAnnotationMessage(text, kind) { var node = byId("annotationMessage"); node.textContent = text || ""; node.className = "message" + (kind ? " " + kind : ""); }
async function saveIntervals() {
  var record = selectedRollout(); if (!record) return;
  normalizeLocalEvents(); byId("saveIntervals").disabled = true; setAnnotationMessage("Saving…", "");
  try {
    var payload = await jsonRequest("/api/annotations/" + encodeURIComponent(record.id), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ events: state.events }) });
    state.events = (payload.events || []).map(function (event) { return Object.assign({}, event); });
    record.annotation_events = state.events.map(function (event) { return Object.assign({}, event); });
    record.annotation_status = state.events.length ? "complete" : "unreviewed";
    state.annotationTarget = payload.annotation_target || state.annotationTarget; state.dirty = false;
    renderIntervals(); renderAnnotateTimeline(); renderRolloutList(); updateHeader();
    setAnnotationMessage("Saved to " + (state.annotationTarget || "annotation file"), "success");
  } catch (error) { setAnnotationMessage(error.message, "error"); }
  finally { byId("saveIntervals").disabled = false; }
}
function updateNavigationButtons() {
  var index = state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  byId("annotatePrevious").disabled = index <= 0; byId("annotateNext").disabled = index < 0 || index >= state.filtered.length - 1;
}
function navigateRollout(delta) {
  var index = state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  var target = state.filtered[index + delta]; if (target) maybeSelectRollout(target.id);
}
async function loadRollouts() {
  var payload = await jsonRequest("/api/rollouts");
  state.rollouts = payload.rollouts || []; state.annotationSource = payload.annotation_source || null; state.annotationTarget = payload.annotation_target || null;
  state.labels = payload.labels || []; LABEL_KEYS = state.labels.map(function (label) { return label.id; });
  populateTaskFilter(); filterRollouts(); updateHeader(); populateResultsRollouts();
  if (state.filtered.length) selectRollout(state.filtered[0].id);
}

function populateResultsRollouts() {
  var select = byId("resultsRollout"); if (!select) return;
  var current = state.results.selectedId;
  select.innerHTML = '<option value="">Select rollout</option>' + state.rollouts.map(function (record) {
    return '<option value="' + escapeHtml(record.id) + '">' + escapeHtml(record.id + " · " + taskLabel(record)) + '</option>';
  }).join("");
  if (current && state.rollouts.some(function (row) { return row.id === current; })) select.value = current;
}
async function loadResultsSources(force) {
  if (state.results.sources.length && !force) return;
  try {
    var payload = await jsonRequest("/api/results/sources");
    state.results.sources = payload.sources || [];
    var select = byId("resultsSource");
    select.innerHTML = '<option value="">No source</option>' + state.results.sources.map(function (source) {
      return '<option value="' + escapeHtml(source.id) + '">' + escapeHtml(source.name) + '</option>';
    }).join("");
    if (state.results.selectedSource && state.results.sources.some(function (row) { return row.id === state.results.selectedSource; })) select.value = state.results.selectedSource;
    else if (state.results.sources.length) { state.results.selectedSource = state.results.sources[0].id; select.value = state.results.selectedSource; }
    if (!state.results.selectedId && state.rollouts.length) { state.results.selectedId = state.rollouts[0].id; byId("resultsRollout").value = state.results.selectedId; }
    if (state.results.selectedSource && state.results.selectedId) await loadResultsCurve();
  } catch (error) { byId("resultsMessage").textContent = error.message; byId("resultsMessage").className = "message error"; }
}
function resultClampFrame(frame) {
  var record = resultRollout(); var max = Math.max(0, Number(record && record.total_frames || 1) - 1);
  return Math.max(0, Math.min(max, Math.round(Number(frame) || 0)));
}
function setupResultsRollout() {
  var record = resultRollout(); if (!record) return;
  state.results.currentFrame = 0;
  byId("resultsEmpty").classList.add("hidden"); byId("resultsContent").classList.remove("hidden");
  byId("resultsTaskTitle").textContent = taskLabel(record);
  byId("resultsRecordMeta").textContent = record.id + " · " + (record.total_frames || "?") + " frames · " + (record.fps || "?") + " fps";
  var cameras = cameraKeys(record); var camera = byId("resultsCamera");
  camera.innerHTML = cameras.map(function (key) { return '<option value="' + escapeHtml(key) + '">' + escapeHtml(key) + '</option>'; }).join(""); camera.value = chooseCamera(record);
  byId("resultsFrameSlider").max = Math.max(0, Number(record.total_frames || 1) - 1); byId("resultsFrameSlider").value = "0";
  setupResultsAnnotationFilters(record);
  renderResultsAnnotationTimeline();
  loadResultsVideo();
}
function resultAnnotationLabels(record) {
  return Array.from(new Set((record && record.annotation_events || []).map(function (event) {
    return Number(event.event_key);
  }).filter(function (key) { return Number.isFinite(key); }))).sort(function (a, b) { return a - b; });
}
function setupResultsAnnotationFilters(record) {
  state.results.annotationLabels = resultAnnotationLabels(record);
  state.results.visibleAnnotationLabels = state.results.annotationLabels.slice();
  renderResultsLabelFilters();
}
function renderResultsLabelFilters() {
  var container = byId("resultsLabelFilters");
  if (!container) return;
  var labels = state.results.annotationLabels || [];
  if (!labels.length) {
    container.innerHTML = '<span class="timeline-help">No labels</span>';
    return;
  }
  var visible = state.results.visibleAnnotationLabels || [];
  container.innerHTML = '<button type="button" class="label-filter-chip' + (visible.length === labels.length ? ' active' : '') + '" data-label-filter="all">All</button>'
    + labels.map(function (key) {
      var label = labelMeta(key); var active = visible.indexOf(key) >= 0;
      return '<button type="button" class="label-filter-chip' + (active ? ' active' : '') + '" data-label-filter="' + key + '"' + (label && active ? ' style="background:' + label.color + ';border-color:' + label.color + '"' : '') + ' title="' + escapeHtml(label ? label.description : 'Unregistered historical label') + '">' + escapeHtml(displayedLabel(key)) + '</button>';
    }).join("");
  container.querySelectorAll("[data-label-filter]").forEach(function (button) {
    button.addEventListener("click", function () {
      var value = button.dataset.labelFilter;
      if (value === "all") {
        state.results.visibleAnnotationLabels = labels.slice();
      } else {
        var key = Number(value);
        var current = state.results.visibleAnnotationLabels.slice();
        var index = current.indexOf(key);
        if (index >= 0) current.splice(index, 1); else current.push(key);
        current.sort(function (a, b) { return a - b; });
        state.results.visibleAnnotationLabels = current;
      }
      renderResultsLabelFilters();
      renderResultsAnnotationTimeline();
    });
  });
}
function renderResultsAnnotationTimeline() {
  var record = resultRollout();
  if (!record) return;
  var visible = state.results.visibleAnnotationLabels || [];
  var events = (record.annotation_events || []).filter(function (event) {
    return visible.indexOf(Number(event.event_key)) >= 0;
  });
  renderPrecisionTimeline("resultsAnnotationTimeline", events, record.total_frames, state.results.currentFrame, null, null, seekResultsFrame, visible);
}
function loadResultsVideo() {
  var record = resultRollout(); if (!record) return;
  var camera = byId("resultsCamera").value || chooseCamera(record); var video = byId("resultsVideo");
  state.results.videoGeneration += 1; var generation = state.results.videoGeneration;
  video.pause(); video.src = "/api/videos/" + encodeURIComponent(record.id) + "?camera=" + encodeURIComponent(camera); video.load();
  if (typeof video.requestVideoFrameCallback === "function") {
    var watch = function (_now, metadata) {
      if (generation !== state.results.videoGeneration) return;
      if (!video.paused && !video.ended) updateResultsFrame(Math.round((metadata.mediaTime || video.currentTime) * (Number(record.fps) || 30)));
      video.requestVideoFrameCallback(watch);
    }; video.requestVideoFrameCallback(watch);
  }
}
function seekResultsFrame(frame) {
  var record = resultRollout(); if (!record) return;
  var value = resultClampFrame(frame); state.results.currentFrame = value;
  byId("resultsVideo").currentTime = value / (Number(record.fps) || 30); updateResultsFrame(value);
}
function updateResultsFrame(frame) {
  var record = resultRollout(); if (!record) return;
  var value = resultClampFrame(frame); state.results.currentFrame = value; byId("resultsFrameSlider").value = String(value);
  byId("resultsFrameText").textContent = "Frame " + value + " / " + Math.max(0, Number(record.total_frames || 1) - 1);
  byId("resultsCurveFrame").textContent = String(value); updateTimelinePlayhead("resultsAnnotationTimeline", value, record.total_frames); updateCurvePlayhead(value); updateResultReadouts(value);
}
function toggleResultsPlay() { var video = byId("resultsVideo"); if (video.paused || video.ended) video.play().catch(function () {}); else video.pause(); }
async function loadResultsCurve() {
  var source = byId("resultsSource").value; var rolloutId = byId("resultsRollout").value;
  state.results.selectedSource = source; state.results.selectedId = rolloutId;
  if (!source || !rolloutId) { byId("resultsContent").classList.add("hidden"); byId("resultsEmpty").classList.remove("hidden"); return; }
  setupResultsRollout();
  var message = byId("resultsMessage"); message.textContent = "Loading model output…"; message.className = "message";
  try {
    var payload = await jsonRequest("/api/results/curve?source=" + encodeURIComponent(source) + "&rollout_id=" + encodeURIComponent(rolloutId));
    state.results.points = payload.points || []; byId("resultsPointCount").textContent = state.results.points.length;
    renderResultCurve(); renderLabelTrack("resultsGtTrack", "label"); renderLabelTrack("resultsPredTrack", "prediction"); updateResultReadouts(state.results.currentFrame);
    message.textContent = state.results.points.length ? payload.source : "No prediction points for this rollout."; message.className = "message";
  } catch (error) { state.results.points = []; renderResultCurve(); message.textContent = error.message; message.className = "message error"; }
}
function renderResultCurve() {
  var svg = byId("resultsCurve").querySelector("svg"); var record = resultRollout(); var points = state.results.points || [];
  if (!record) { svg.innerHTML = ""; return; }
  var total = Math.max(1, Number(record.total_frames) || 1); var left = 42, right = 985, top = 16, bottom = 334, width = right - left, height = bottom - top;
  var html = "";
  [0, .25, .5, .75, 1].forEach(function (p) { var y = bottom - p * height; html += '<line class="curve-grid" x1="' + left + '" y1="' + y + '" x2="' + right + '" y2="' + y + '"></line><text class="curve-axis-label" x="4" y="' + (y + 6) + '">' + p.toFixed(2) + '</text>'; });
  [0, .25, .5, .75, 1].forEach(function (p) { var x = left + p * width; html += '<line class="curve-grid" x1="' + x + '" y1="' + top + '" x2="' + x + '" y2="' + bottom + '"></line><text class="curve-axis-label" x="' + (x - 12) + '" y="356">' + Math.round((total - 1) * p) + '</text>'; });
  [0,1,2].forEach(function (classIndex) {
    var path = points.map(function (point, index) {
      var x = left + timelinePercent(point.frame, total) / 100 * width; var probability = Number(point.probabilities && point.probabilities[classIndex]) || 0; var y = bottom - probability * height;
      return (index ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
    }).join(" ");
    if (path) html += '<path class="curve-path p' + classIndex + '" d="' + path + '"></path>';
  });
  html += '<line id="resultsCurvePlayhead" class="curve-playhead" x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + bottom + '"></line>';
  html += '<circle id="resultsCurvePoint" class="curve-point-current" cx="' + left + '" cy="' + bottom + '" r="5"></circle>';
  svg.innerHTML = html;
  svg.onclick = function (event) { var box = svg.getBoundingClientRect(); var ratio = Math.max(0, Math.min(1, (event.clientX - box.left) / Math.max(1, box.width))); seekResultsFrame(Math.round(ratio * (total - 1))); };
  updateCurvePlayhead(state.results.currentFrame);
}
function currentResultPoint(frame) {
  var points = state.results.points || []; if (!points.length) return null;
  var current = points[0];
  for (var i = 0; i < points.length; i += 1) { if (points[i].frame <= frame) current = points[i]; else break; }
  return current;
}
function updateCurvePlayhead(frame) {
  var record = resultRollout(); if (!record) return;
  var line = byId("resultsCurvePlayhead"); var dot = byId("resultsCurvePoint"); if (!line || !dot) return;
  var left = 42, right = 985, top = 16, bottom = 334, width = right - left, height = bottom - top; var x = left + timelinePercent(frame, record.total_frames) / 100 * width;
  line.setAttribute("x1", x); line.setAttribute("x2", x);
  var point = currentResultPoint(frame); var probability = point && point.probabilities ? Math.max.apply(null, point.probabilities.map(Number)) : 0; dot.setAttribute("cx", x); dot.setAttribute("cy", bottom - probability * height);
}
function renderLabelTrack(containerId, key) {
  var container = byId(containerId); var record = resultRollout(); var points = state.results.points || []; if (!container || !record) return;
  var total = Math.max(1, Number(record.total_frames) || 1); var html = "";
  points.forEach(function (point, index) {
    var start = point.frame; var end = index + 1 < points.length ? Math.max(start, points[index + 1].frame - 1) : Math.min(total - 1, start + Math.max(1, index ? start - points[index - 1].frame : 1));
    var left = timelinePercent(start, total), right = timelinePercent(end, total), width = Math.max(.3, right - left + 100 / total); var value = Number(point[key]);
    html += '<div class="label-block label-' + value + '" style="left:' + left + '%;width:' + Math.min(width,100-left) + '%"><span>' + value + '</span></div>';
  });
  container.innerHTML = html;
}
function updateResultReadouts(frame) {
  var point = currentResultPoint(frame); byId("resultsGtReadout").textContent = point ? String(point.label) : "—"; byId("resultsPredReadout").textContent = point ? String(point.prediction) : "—";
}

async function loadAnalysis() {
  var cards = byId("analysisCards"); cards.innerHTML = '<div class="metric-card"><span class="muted">Loading…</span></div>';
  try {
    var data = await jsonRequest("/api/analysis");
    cards.innerHTML = [["Rollouts",data.rollouts],["Annotated",data.reviewed_rollouts],["Coverage",formatPercent(data.coverage)],["Intervals",data.events]].map(function (item) { return '<div class="metric-card"><div class="eyebrow">' + escapeHtml(item[0]) + '</div><div class="metric-value">' + escapeHtml(item[1]) + '</div></div>'; }).join("");
    byId("eventAnalysisTable").innerHTML = '<table class="data-table"><thead><tr><th>Label</th><th>Count</th><th>Mean duration</th></tr></thead><tbody>' + (data.event_types || []).map(function (row) { return '<tr><td><strong>' + row.event_key + ' · ' + escapeHtml(row.name || labelName(row.event_key)) + '</strong><div class="muted">' + escapeHtml(row.description || '') + '</div></td><td>' + row.count + '</td><td>' + (row.mean_duration_frames == null ? '—' : row.mean_duration_frames.toFixed(1) + ' f') + '</td></tr>'; }).join("") + '</tbody></table>';
    byId("taskAnalysisTable").innerHTML = '<table class="data-table"><thead><tr><th>Task</th><th>Rollouts</th><th>Annotated</th><th>Coverage</th><th>Intervals</th></tr></thead><tbody>' + (data.tasks || []).map(function (row) { return '<tr><td>' + escapeHtml(row.task) + '</td><td>' + row.rollouts + '</td><td>' + row.reviewed + '</td><td>' + formatPercent(row.rollouts ? row.reviewed / row.rollouts : 0) + '</td><td>' + row.events + '</td></tr>'; }).join("") + '</tbody></table>';
  } catch (error) { cards.innerHTML = '<div class="metric-card"><span class="message error">' + escapeHtml(error.message) + '</span></div>'; }
}
async function loadRuns() {
  var container = byId("runsList"); container.innerHTML = '<div class="muted">Loading tactile experiment outputs…</div>';
  try {
    var payload = await jsonRequest("/api/runs"); byId("runsRoot").textContent = "Scanning: " + payload.runs_root; var runs = payload.runs || [];
    if (!runs.length) { container.innerHTML = '<div class="muted" style="padding:18px 0">No SHARPA output directories found.</div>'; return; }
    container.innerHTML = runs.map(function (run) { return '<div class="run-row"><div><div class="run-path">' + escapeHtml(run.path) + '</div><div class="muted" style="font-size:.57rem;margin-top:3px">' + escapeHtml(formatDate(run.modified_at)) + '</div></div><div class="run-markers">' + (run.markers || []).map(function (marker) { return badge(marker, ""); }).join("") + '</div><span class="badge">tactile</span></div>'; }).join("");
  } catch (error) { container.innerHTML = '<div class="message error">' + escapeHtml(error.message) + '</div>'; }
}
function renderLabelSettings() {
  var container = byId("labelRegistryRows");
  if (!container) return;
  var rows = state.labelDraft || [];
  var existing = state.labels.map(function (label) { return label.id; });
  container.innerHTML = rows.map(function (label) {
    var isNew = existing.indexOf(label.id) < 0;
    return '<div class="label-registry-row" data-label-id="' + label.id + '">'
      + '<div><span class="label-field-title">ID</span><strong class="label-id">' + label.id + '</strong></div>'
      + '<label><span>Name</span><input type="text" maxlength="64" data-field="name" value="' + escapeHtml(label.name) + '"></label>'
      + '<label><span>Description</span><textarea rows="2" maxlength="500" data-field="description">' + escapeHtml(label.description) + '</textarea></label>'
      + '<label><span>Color</span><input type="color" data-field="color" value="' + escapeHtml(label.color) + '"></label>'
      + '<label><span>Eligibility</span>' + (isNew
        ? '<select data-field="scope"><option value="all"' + (label.scope === 'all' ? ' selected' : '') + '>All</option><option value="failure"' + (label.scope === 'failure' ? ' selected' : '') + '>Failure only</option></select>'
        : '<span class="label-scope-static">' + (label.scope === 'failure' ? 'Failure only' : 'All rollouts') + '</span>') + '</label>'
      + '<label><span>Active</span><input type="checkbox" data-field="active"' + (label.active ? ' checked' : '') + '></label>'
      + '</div>';
  }).join('');
}
function updateLabelDraft(event) {
  var target = event.target;
  var field = target && target.dataset && target.dataset.field;
  var row = target && target.closest && target.closest('[data-label-id]');
  if (!field || !row || !state.labelDraft) return;
  var label = state.labelDraft.find(function (item) { return item.id === Number(row.dataset.labelId); });
  if (!label) return;
  if (field === 'active') label[field] = Boolean(target.checked);
  else if (['name', 'description', 'color', 'scope'].indexOf(field) >= 0) label[field] = target.value;
}
function addLabelDefinition() {
  if (!state.labelDraft) return;
  var next = Math.max(0, ...state.labelDraft.map(function (label) { return label.id; })) + 1;
  state.labelDraft.push({ id: next, name: 'New label', description: '', color: '#64748b', scope: 'all', active: true });
  renderLabelSettings();
  var node = byId("labelRegistryMessage"); node.textContent = 'Configure the new ID and save. IDs are never reused.'; node.className = 'message';
}
async function loadLabelSettings() {
  try {
    var data = await jsonRequest('/api/labels');
    state.labels = data.labels || [];
    LABEL_KEYS = state.labels.map(function (label) { return label.id; });
    state.labelDraft = state.labels.map(function (label) { return Object.assign({}, label); });
    renderLabelSettings();
  } catch (error) {
    byId('labelRegistryRows').innerHTML = '<div class="message error">' + escapeHtml(error.message) + '</div>';
  }
}
async function saveLabelDefinitions() {
  if (!state.labelDraft) return;
  var button = byId('saveLabelDefinitions'), message = byId('labelRegistryMessage');
  button.disabled = true;
  try {
    var data = await jsonRequest('/api/labels', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema_version: 1, dataset: 'failrecovery', labels: state.labelDraft })
    });
    state.labels = data.labels || [];
    LABEL_KEYS = state.labels.map(function (label) { return label.id; });
    state.labelDraft = state.labels.map(function (label) { return Object.assign({}, label); });
    renderLabelSettings();
    if (selectedRollout()) { renderIntervals(); renderAnnotateTimeline(); }
    if (resultRollout()) { renderResultsLabelFilters(); renderResultsAnnotationTimeline(); }
    message.textContent = 'Label definitions saved. Event IDs and saved intervals are unchanged.';
    message.className = 'message success';
  } catch (error) {
    message.textContent = error.message;
    message.className = 'message error';
  } finally {
    button.disabled = false;
  }
}

async function loadSettings() {
  try {
    var payload = await jsonRequest("/api/settings"); var settings = payload.settings || {}; state.settings = settings; applyTheme(settings.theme);
    byId("settingsSourceRoot").value = settings.source_project_root || "."; byId("settingsManifest").value = payload.source_project_root_resolved ? payload.source_project_root_resolved + "/" + (payload.manifest_path || "") : (payload.manifest_path || "");
    byId("settingsAnnotations").value = settings.annotations_path || ""; byId("settingsSeedGlob").value = settings.annotation_seed_glob || ""; byId("settingsRunsRoot").value = settings.runs_root || ""; byId("settingsCamera").value = settings.default_camera || "cam_high"; byId("settingsKind").value = settings.default_tactile_kind || "deform"; byId("settingsTheme").value = settings.theme || "light"; byId("settingsSource").textContent = payload.annotation_source || "None";
    await loadLabelSettings();
  } catch (error) { var node = byId("settingsMessage"); node.textContent = error.message; node.className = "message error"; }
}
async function saveSettings(event) {
  event.preventDefault(); var payload = { source_project_root: byId("settingsSourceRoot").value.trim(), annotations_path: byId("settingsAnnotations").value.trim(), annotation_seed_glob: byId("settingsSeedGlob").value.trim(), runs_root: byId("settingsRunsRoot").value.trim(), default_camera: byId("settingsCamera").value.trim(), default_tactile_kind: byId("settingsKind").value, theme: byId("settingsTheme").value }; var message = byId("settingsMessage");
  try { var result = await jsonRequest("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); state.settings = result.settings; applyTheme(state.settings.theme); message.textContent = "Settings saved."; message.className = "message success"; await loadSettings(); await loadResultsSources(true); }
  catch (error) { message.textContent = error.message; message.className = "message error"; }
}

function setVideoHealth(name, text, kind, detail) {
  state.videoDiagnostics[name] = { text: text, kind: kind, detail: detail || "", updated_at: new Date().toISOString() };
  var node = byId(name === "annotate" ? "annotateVideoHealth" : "resultsVideoHealth");
  if (node) { node.textContent = text; node.className = "health-pill " + (kind || "idle"); }
}
function monitorVideo(videoId, name) {
  var video = byId(videoId); var watchdog = null; var loadStarted = 0;
  function emit(eventName, level, extra) {
    var payload = Object.assign({ event: eventName, level: level || "info", component: "video", video: name, src: video.currentSrc || video.src || "", ready_state: video.readyState, network_state: video.networkState, current_time: video.currentTime }, extra || {});
    queueTelemetry(payload);
  }
  video.addEventListener("loadstart", function () { loadStarted = performance.now(); setVideoHealth(name, "Video loading", "loading", "loadstart"); emit("loadstart"); if (watchdog) clearTimeout(watchdog); watchdog = setTimeout(function () { if (video.readyState < 1) { setVideoHealth(name, "No metadata · check network/server", "error", "metadata timeout"); emit("watchdog_timeout", "error", { elapsed_ms: performance.now() - loadStarted }); } }, 8000); });
  video.addEventListener("loadedmetadata", function () { if (watchdog) clearTimeout(watchdog); setVideoHealth(name, "Metadata ready", "ready", video.videoWidth + "×" + video.videoHeight); emit("loadedmetadata", "info", { elapsed_ms: loadStarted ? performance.now() - loadStarted : null, duration: video.duration, width: video.videoWidth, height: video.videoHeight }); });
  video.addEventListener("canplay", function () { setVideoHealth(name, "Video ready", "ready", "canplay"); emit("canplay"); });
  video.addEventListener("playing", function () { setVideoHealth(name, "Playing", "ready", "playing"); emit("playing"); });
  video.addEventListener("waiting", function () { setVideoHealth(name, "Buffering", "buffering", "waiting"); emit("waiting", "error"); });
  video.addEventListener("stalled", function () { setVideoHealth(name, "Network stalled", "error", "stalled"); emit("stalled", "error"); });
  video.addEventListener("error", function () { var code = video.error ? video.error.code : null; setVideoHealth(name, "Video error " + (code || ""), "error", video.currentSrc || video.src); emit("error", "error", { media_error_code: code }); });
  video.addEventListener("seeking", function () { emit("seeking"); }); video.addEventListener("seeked", function () { emit("seeked"); });
}
async function loadDiagnostics() {
  try {
    var data = await jsonRequest("/api/diagnostics"); var videoLatency = (data.latency_by_resource_ms || {}).video || {}; var errors = (data.server_errors || []).length + (data.client_errors || []).length; var tactileCache = data.tactile_cache || {}; var episodeCache = tactileCache.episodes || {}; var seriesCache = tactileCache.series || {};
    byId("diagnosticKpis").innerHTML = [['Requests',data.requests || 0],['Video p50',videoLatency.p50 == null ? '—' : Math.round(videoLatency.p50) + ' ms'],['Video p95',videoLatency.p95 == null ? '—' : Math.round(videoLatency.p95) + ' ms'],['Errors',errors],['Episode cache',(episodeCache.size || 0) + ' / ' + (episodeCache.limit || '—')],['Series cache',(seriesCache.size || 0) + ' / ' + (seriesCache.limit || '—')],['Cache evict',(episodeCache.evictions || 0) + ' / ' + (seriesCache.evictions || 0)],['Video req',(data.resource_counts || {}).video || 0]].map(function (item) { return '<div class="diagnostic-kpi"><span class="eyebrow">' + escapeHtml(item[0]) + '</span><strong>' + escapeHtml(item[1]) + '</strong></div>'; }).join("");
    var videos = Object.keys(state.videoDiagnostics).map(function (key) { var row = state.videoDiagnostics[key]; return key + ': ' + row.text + (row.detail ? ' · ' + row.detail : '') + ' · ' + row.updated_at; }); byId("diagnosticVideo").textContent = videos.length ? videos.join("\n") : "No video events yet.";
    var errorRows = (data.server_errors || []).slice(-6).map(function (row) { return 'server ' + (row.status || '') + ' ' + (row.path || '') + ' · ' + Math.round(row.duration_ms || 0) + 'ms'; }).concat((data.client_errors || []).slice(-6).map(function (row) { return 'client ' + (row.event || '') + ' · ' + (row.message || row.src || row.url || ''); })); byId("diagnosticErrors").textContent = errorRows.length ? errorRows.join("\n") : "None.";
    var paths = data.log_paths || {}; byId("diagnosticPaths").textContent = 'server: ' + (paths.server || '—') + '\nclient: ' + (paths.client || '—');
  } catch (error) { byId("diagnosticErrors").textContent = error.message; }
}
async function checkHealth() {
  try { var started = performance.now(); var response = await fetch("/api/health", { cache: "no-store" }); if (!response.ok) throw new Error("HTTP " + response.status); await response.json(); setGlobalStatus(true, "Server online · " + Math.round(performance.now() - started) + " ms"); }
  catch (error) { setGlobalStatus(false, "Server unavailable"); queueTelemetry({ event: "health_error", level: "error", message: String(error && error.message || error) }); }
}
function initPerformanceObserver() {
  if (!("PerformanceObserver" in window)) return;
  var tactileCounter = 0;
  try {
    var observer = new PerformanceObserver(function (list) {
      list.getEntries().forEach(function (entry) {
        var isVideo = entry.name.indexOf("/api/videos/") >= 0;
        var isTactile = entry.name.indexOf("/api/tactile/") >= 0;
        if (!isVideo && !isTactile) return;
        if (isTactile) { tactileCounter += 1; if (entry.duration < 250 && tactileCounter % 30 !== 0) return; }
        queueTelemetry({ event: "resource_timing", level: entry.duration >= 1000 ? "error" : "info", resource: isVideo ? "video" : "tactile", url: entry.name, duration_ms: entry.duration, transfer_size: entry.transferSize || 0, encoded_size: entry.encodedBodySize || 0, decoded_size: entry.decodedBodySize || 0 });
      });
    }); observer.observe({ type: "resource", buffered: true });
  } catch (_error) {}
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach(function (button) { button.addEventListener("click", function () { switchView(button.dataset.view); }); });
  byId("annotateSearch").addEventListener("input", filterRollouts); byId("annotateTaskFilter").addEventListener("change", filterRollouts); byId("annotateReviewFilter").addEventListener("change", filterRollouts);
  byId("annotatePrevious").addEventListener("click", function () { navigateRollout(-1); }); byId("annotateNext").addEventListener("click", function () { navigateRollout(1); });
  byId("annotateCamera").addEventListener("change", function () { state.tactileSeries = null; state.tactileSeriesKey = ""; state.tactileSeriesPromise = null; state.tactileAppliedKey = ""; state.tactilePendingFrame = null; state.tactileGeneration += 1; loadAnnotateVideo(); seekFrame(0); }); byId("annotatePlay").addEventListener("click", togglePlay); byId("annotateStepBack").addEventListener("click", function () { seekFrame(state.currentFrame - 1); }); byId("annotateStepForward").addEventListener("click", function () { seekFrame(state.currentFrame + 1); }); byId("annotateFrameSlider").addEventListener("input", function (event) { seekFrame(event.target.value); });
  byId("annotateVideo").addEventListener("play", function () { byId("annotatePlay").textContent = "Pause"; }); byId("annotateVideo").addEventListener("pause", function () { byId("annotatePlay").textContent = "Play"; }); byId("annotateVideo").addEventListener("timeupdate", function () { var record = selectedRollout(); if (record) updateFrameUi(Math.round(byId("annotateVideo").currentTime * (Number(record.fps) || 30)), false); });
  byId("addInterval").addEventListener("click", function () {
    var label = state.labels.find(function (item) { return eligibleLabel(item, selectedRollout()); });
    if (label) addInterval(label.id); else setAnnotationMessage("No active labels for this rollout.", "error");
  }); byId("saveIntervals").addEventListener("click", saveIntervals);
  byId("addLabelDefinition").addEventListener("click", addLabelDefinition);
  byId("saveLabelDefinitions").addEventListener("click", saveLabelDefinitions);
  byId("labelRegistryRows").addEventListener("input", updateLabelDraft);
  byId("labelRegistryRows").addEventListener("change", updateLabelDraft);
  byId("resultsSource").addEventListener("change", loadResultsCurve); byId("resultsRollout").addEventListener("change", loadResultsCurve); byId("resultsCamera").addEventListener("change", function () { loadResultsVideo(); seekResultsFrame(0); }); byId("refreshResults").addEventListener("click", function () { loadResultsSources(true); }); byId("resultsPlay").addEventListener("click", toggleResultsPlay); byId("resultsStepBack").addEventListener("click", function () { seekResultsFrame(state.results.currentFrame - 1); }); byId("resultsStepForward").addEventListener("click", function () { seekResultsFrame(state.results.currentFrame + 1); }); byId("resultsFrameSlider").addEventListener("input", function (event) { seekResultsFrame(event.target.value); }); byId("resultsVideo").addEventListener("play", function () { byId("resultsPlay").textContent = "Pause"; }); byId("resultsVideo").addEventListener("pause", function () { byId("resultsPlay").textContent = "Play"; }); byId("resultsVideo").addEventListener("timeupdate", function () { var record = resultRollout(); if (record) updateResultsFrame(Math.round(byId("resultsVideo").currentTime * (Number(record.fps) || 30))); });
  byId("refreshRuns").addEventListener("click", loadRuns); byId("refreshAnalysis").addEventListener("click", loadAnalysis); byId("settingsForm").addEventListener("submit", saveSettings); byId("refreshDiagnostics").addEventListener("click", loadDiagnostics);
  window.addEventListener("beforeunload", function (event) { flushTelemetry(); if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("keydown", function (event) {
    if (state.view !== "annotate" || isTypingTarget(event.target)) return;
    if (event.key === " ") { event.preventDefault(); togglePlay(); return; } if (event.key === ",") { event.preventDefault(); navigateRollout(-1); return; } if (event.key === ".") { event.preventDefault(); navigateRollout(1); return; }
    if (event.key === "ArrowLeft") { event.preventDefault(); seekFrame(state.currentFrame - 1); return; } if (event.key === "ArrowRight") { event.preventDefault(); seekFrame(state.currentFrame + 1); return; }
    if (event.key === "ArrowUp" && state.events.length) { event.preventDefault(); state.activeEvent = state.activeEvent == null ? 0 : Math.max(0,state.activeEvent-1); renderIntervals(); renderAnnotateTimeline(); return; }
    if (event.key === "ArrowDown" && state.events.length) { event.preventDefault(); state.activeEvent = state.activeEvent == null ? 0 : Math.min(state.events.length-1,state.activeEvent+1); renderIntervals(); renderAnnotateTimeline(); return; }
    if (["1","2","3","4"].indexOf(event.key) >= 0) { event.preventDefault(); setActiveEventType(Number(event.key)); return; } if (event.key.toLowerCase() === "q") { event.preventDefault(); setActiveBoundary("start"); return; } if (event.key.toLowerCase() === "w") { event.preventDefault(); setActiveBoundary("end"); }
  });
}
async function init() {
  bindEvents(); monitorVideo("annotateVideo", "annotate"); monitorVideo("resultsVideo", "results"); initPerformanceObserver();
  var requested = (location.hash || "#annotate").slice(1); if (!byId("view-" + requested)) requested = "annotate"; switchView(requested, false);
  try {
    var settingsPayload = await jsonRequest("/api/settings"); state.settings = settingsPayload.settings || {}; state.annotationSource = settingsPayload.annotation_source || null; state.annotationTarget = settingsPayload.annotation_target || null; applyTheme(state.settings.theme); await loadRollouts(); await checkHealth();
  } catch (error) { setGlobalStatus(false, error.message); byId("datasetStatus").textContent = "Dataset unavailable"; }
  window.setInterval(checkHealth, 15000); window.setInterval(function () { if (state.view === "settings") loadDiagnostics(); }, 10000);
}

init();
