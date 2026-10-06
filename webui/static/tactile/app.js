"use strict";

(function installStandaloneTactileReview() {
  var FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
  var LABELS = {
    thumb: "Thumb",
    index: "Index",
    middle: "Middle",
    ring: "Ring",
    pinky: "Pinky"
  };
  var PREFERRED_CAMERAS = ["cam_high", "cam_wrist", "cam_left_wrist", "cam_right_wrist"];
  var state = {
    rollouts: [],
    filtered: [],
    selectedId: "",
    camera: "",
    series: null,
    seriesKey: "",
    loadingSeriesKey: "",
    loadingSeriesPromise: null,
    seriesSerial: 0,
    eventLookup: null,
    currentFrame: 0,
    lastSyncKey: "",
    appliedSpriteKey: "",
    requestedSpriteKey: "",
    spriteSerial: 0,
    spritePreloads: new Map(),
    videoFrameCallbackId: 0,
    animationFrameId: 0
  };

  function byId(id) { return document.getElementById(id); }
  var video = byId("rolloutVideo");
  var frameSlider = byId("frameSlider");
  var grid = byId("tactileGrid");
  var curveNode = byId("tactileCurve");
  var statusNode = byId("tactileStatus");

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function supportsTactile(record) {
    return Boolean(
      record
      && record.synchronized_frames_path
      && record.tactile_events_path
      && record.tactile_stream_paths
    );
  }

  function selectedRecord() {
    return state.rollouts.find(function (record) { return record.id === state.selectedId; }) || null;
  }

  function effectiveOutcome(record) {
    return record && record.annotation && record.annotation.outcome_label
      ? record.annotation.outcome_label
      : (record ? record.ground_truth_outcome : "");
  }

  function badge(value, extraClass) {
    var text = String(value || "").replace(/_/g, " ");
    return '<span class="badge ' + escapeHtml(extraClass || "") + '">' + escapeHtml(text) + '</span>';
  }

  function formatTime(seconds) {
    var value = Math.max(0, Number(seconds) || 0);
    var minutes = Math.floor(value / 60);
    var remainder = value - minutes * 60;
    return String(minutes).padStart(2, "0") + ":" + remainder.toFixed(3).padStart(6, "0");
  }

  function cameraLabel(key) {
    return String(key || "")
      .replace(/^cam_/, "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, function (match) { return match.toUpperCase(); });
  }

  function cameraKeys(record) {
    var paths = record && record.camera_video_paths;
    if (!paths || typeof paths !== "object") return [];
    var keys = Object.keys(paths).filter(function (key) {
      return typeof paths[key] === "string" && paths[key].trim().length > 0;
    });
    keys.sort(function (left, right) {
      var leftIndex = PREFERRED_CAMERAS.indexOf(left);
      var rightIndex = PREFERRED_CAMERAS.indexOf(right);
      if (leftIndex < 0) leftIndex = PREFERRED_CAMERAS.length;
      if (rightIndex < 0) rightIndex = PREFERRED_CAMERAS.length;
      return leftIndex - rightIndex || left.localeCompare(right);
    });
    return keys;
  }

  function renderRolloutList() {
    var query = String(byId("searchInput").value || "").trim().toLowerCase();
    state.filtered = state.rollouts.filter(function (record) {
      var haystack = [
        record.id,
        record.task_description,
        record.task_suite,
        record.task_id,
        record.manifest_label,
        record.manifest_source
      ].join(" ").toLowerCase();
      return !query || haystack.indexOf(query) >= 0;
    });
    byId("visibleCount").textContent = state.filtered.length;
    var list = byId("rolloutList");
    if (!state.filtered.length) {
      list.innerHTML = '<div class="placeholder">No tactile rollouts match this search.</div>';
      updateNavigationButtons();
      return;
    }
    list.innerHTML = state.filtered.map(function (record) {
      var outcome = effectiveOutcome(record) || "unknown";
      var active = record.id === state.selectedId ? " active" : "";
      var title = record.task_description || (String(record.task_suite || "task") + " " + String(record.task_id || ""));
      var source = record.manifest_label || record.manifest_source || "manifest";
      return '<button class="rollout-item' + active + '" data-rollout-id="' + escapeHtml(record.id) + '" type="button">'
        + '<div class="badge-row">' + badge(outcome, outcome) + '</div>'
        + '<div class="rollout-title">' + escapeHtml(title) + '</div>'
        + '<div class="rollout-meta">' + escapeHtml(record.id) + '<br>'
        + escapeHtml(source) + ' · ' + escapeHtml(record.total_frames) + 'f</div>'
        + '</button>';
    }).join("");
    updateNavigationButtons();
  }

  function updateListSelection() {
    byId("rolloutList").querySelectorAll("[data-rollout-id]").forEach(function (button) {
      button.classList.toggle("active", button.dataset.rolloutId === state.selectedId);
    });
  }

  function filteredIndex() {
    return state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  }

  function updateNavigationButtons() {
    var index = filteredIndex();
    byId("previousButton").disabled = index <= 0;
    byId("nextButton").disabled = index < 0 || index >= state.filtered.length - 1;
  }

  function navigate(delta) {
    var index = filteredIndex();
    var target = state.filtered[index + delta];
    if (target) selectRollout(target.id);
  }

  function setHeader(record) {
    var outcome = effectiveOutcome(record) || "unknown";
    byId("recordBadges").innerHTML = badge(outcome, outcome)
      + badge(record.dataset_role || record.task_suite || "tactile", "");
    byId("taskTitle").textContent = record.task_description || record.id;
    byId("recordMeta").textContent = record.id
      + " · task " + String(record.task_id == null ? "—" : record.task_id)
      + " · episode " + String(record.episode_index == null ? "—" : record.episode_index)
      + " · " + String(record.fps || "—") + " fps"
      + " · " + String(record.total_frames || "—") + " frames";
  }

  function populateCameraSelect(record, requested) {
    var keys = cameraKeys(record);
    var select = byId("cameraSelect");
    select.innerHTML = keys.map(function (key) {
      return '<option value="' + escapeHtml(key) + '">' + escapeHtml(cameraLabel(key)) + '</option>';
    }).join("");
    var fallback = keys.indexOf(record.observation_key) >= 0 ? record.observation_key : keys[0];
    var target = keys.indexOf(requested) >= 0 ? requested : fallback;
    state.camera = target || "cam_high";
    select.value = state.camera;
  }

  function setVideoSource(record, preserveFrame) {
    var fps = Number(record.fps);
    var frame = preserveFrame ? state.currentFrame : 0;
    state.currentFrame = Math.max(0, Math.min(Number(record.total_frames || 1) - 1, frame));
    video.pause();
    video.dataset.rolloutId = record.id;
    video.dataset.videoView = state.camera;
    video.src = "/api/videos/" + encodeURIComponent(record.id)
      + "?camera=" + encodeURIComponent(state.camera)
      + "&v=tactile-standalone";
    video.addEventListener("loadedmetadata", function onMetadata() {
      if (Number.isFinite(fps) && fps > 0) {
        try { video.currentTime = state.currentFrame / fps; } catch (_error) {}
      }
      updateReadout(state.currentFrame);
      refreshTactile(state.currentFrame, true);
    }, { once: true });
    video.load();
    byId("playButton").textContent = "Play";
  }

  function resetTactileState() {
    state.series = null;
    state.seriesKey = "";
    state.loadingSeriesKey = "";
    state.loadingSeriesPromise = null;
    state.seriesSerial += 1;
    state.eventLookup = null;
    state.lastSyncKey = "";
    state.appliedSpriteKey = "";
    state.requestedSpriteKey = "";
    state.spriteSerial += 1;
    grid.innerHTML = "";
    curveNode.innerHTML = '<div class="placeholder">Loading f6 history…</div>';
    statusNode.textContent = "Loading tactile timeline…";
  }

  async function selectRollout(id) {
    var record = state.rollouts.find(function (item) { return item.id === id; });
    if (!record) return;
    state.selectedId = id;
    state.currentFrame = 0;
    setHeader(record);
    populateCameraSelect(record, state.camera);
    frameSlider.max = String(Math.max(0, Number(record.total_frames || 1) - 1));
    frameSlider.value = "0";
    byId("emptyState").classList.add("hidden");
    byId("viewerContent").classList.remove("hidden");
    updateListSelection();
    updateNavigationButtons();
    resetTactileState();
    setVideoSource(record, false);
    await ensureSeries(record, state.camera);
    refreshTactile(0, true);
  }

  function frameFromVideo(record, mediaTime) {
    var fps = Number(record && record.fps);
    var time = Number.isFinite(Number(mediaTime)) ? Number(mediaTime) : Number(video.currentTime);
    if (!Number.isFinite(fps) || fps <= 0 || !Number.isFinite(time)) return state.currentFrame;
    var maxFrame = Math.max(0, Number(record.total_frames || 1) - 1);
    return Math.max(0, Math.min(maxFrame, Math.round(time * fps)));
  }

  function updateReadout(frame) {
    var record = selectedRecord();
    if (!record) return;
    var maxFrame = Math.max(0, Number(record.total_frames || 1) - 1);
    state.currentFrame = Math.max(0, Math.min(maxFrame, Math.round(Number(frame) || 0)));
    frameSlider.value = String(state.currentFrame);
    byId("frameReadout").textContent = "Frame " + state.currentFrame + " / " + maxFrame;
    byId("timeReadout").textContent = formatTime(video.currentTime);
    updateCurvePlayhead(state.currentFrame);
  }

  function seekFrame(frame) {
    var record = selectedRecord();
    if (!record) return;
    var maxFrame = Math.max(0, Number(record.total_frames || 1) - 1);
    var target = Math.max(0, Math.min(maxFrame, Math.round(Number(frame) || 0)));
    var fps = Number(record.fps);
    state.currentFrame = target;
    if (Number.isFinite(fps) && fps > 0) {
      try { video.currentTime = target / fps; } catch (_error) {}
    }
    updateReadout(target);
    refreshTactile(target, true);
  }

  function buildEventLookup(series) {
    var lookup = {};
    FINGERS.forEach(function (finger) {
      lookup[finger] = new Map();
      var rows = series && series.fingers && Array.isArray(series.fingers[finger])
        ? series.fingers[finger] : [];
      rows.forEach(function (row) {
        if (row && row.event_id != null) lookup[finger].set(String(row.event_id), row);
      });
    });
    return lookup;
  }

  function nearestSyncFrame(frame) {
    var rows = state.series && Array.isArray(state.series.sync_frames) ? state.series.sync_frames : [];
    if (!rows.length) return null;
    var low = 0;
    var high = rows.length - 1;
    while (low <= high) {
      var mid = (low + high) >> 1;
      var value = Number(rows[mid].frame);
      if (value < frame) low = mid + 1;
      else if (value > frame) high = mid - 1;
      else return { row: rows[mid], index: mid };
    }
    if (low <= 0) return { row: rows[0], index: 0 };
    if (low >= rows.length) return { row: rows[rows.length - 1], index: rows.length - 1 };
    var before = rows[low - 1];
    var after = rows[low];
    return Math.abs(frame - Number(before.frame)) <= Math.abs(Number(after.frame) - frame)
      ? { row: before, index: low - 1 }
      : { row: after, index: low };
  }

  function currentFingerData(syncRow, finger) {
    var sync = syncRow && syncRow.fingers && syncRow.fingers[finger] ? syncRow.fingers[finger] : {};
    var eventId = sync.event_id;
    var event = state.eventLookup && state.eventLookup[finger] && eventId != null
      ? state.eventLookup[finger].get(String(eventId)) : null;
    if (!event && eventId == null) return null;
    event = event || {};
    return {
      event_id: eventId,
      f6: event.f6,
      valid: sync.valid == null ? event.valid : sync.valid,
      stale: sync.stale,
      age_ms: sync.age_ms,
      image_kinds: event.image_kinds || []
    };
  }

  function formatF6(value) {
    if (!Array.isArray(value) || !value.length) return '<span class="empty">No f6</span>';
    return value.slice(0, 6).map(function (item, index) {
      var number = Number(item);
      var text = Number.isFinite(number) ? number.toFixed(3) : "—";
      return '<span><b>f' + (index + 1) + '</b>' + escapeHtml(text) + '</span>';
    }).join("");
  }

  function renderFinger(finger, data, kind) {
    if (!data) {
      return '<article class="finger-card"><header><strong>' + escapeHtml(LABELS[finger])
        + '</strong><span class="chip">missing</span></header><div class="tactile-image placeholder">No sample</div></article>';
    }
    var kinds = Array.isArray(data.image_kinds) ? data.image_kinds : [];
    var hasImage = kinds.indexOf(kind) >= 0 && data.event_id != null;
    var invalid = data.valid === false;
    var stale = data.stale === true;
    var age = Number(data.age_ms);
    var fingerIndex = FINGERS.indexOf(finger);
    var image = hasImage
      ? '<div class="tactile-image tactile-sprite finger-' + fingerIndex + '" role="img" aria-label="'
        + escapeHtml(LABELS[finger]) + ' tactile ' + escapeHtml(kind) + '"></div>'
      : '<div class="tactile-image placeholder">No image</div>';
    return '<article class="finger-card' + (invalid ? ' is-invalid' : '') + (stale ? ' is-stale' : '') + '">'
      + '<header><strong>' + escapeHtml(LABELS[finger]) + '</strong><span class="chips">'
      + '<span class="chip ' + (invalid ? 'invalid' : '') + '">' + (invalid ? 'invalid' : 'valid') + '</span>'
      + (stale ? '<span class="chip stale">stale</span>' : '')
      + '</span></header>'
      + image
      + '<div class="finger-meta">event ' + escapeHtml(data.event_id == null ? "—" : data.event_id)
      + (Number.isFinite(age) ? ' · ' + age.toFixed(1) + ' ms' : '') + '</div>'
      + '<div class="f6">' + formatF6(data.f6) + '</div></article>';
  }

  function spriteUrl(record, camera, frame, kind) {
    return "/api/tactile/" + encodeURIComponent(record.id) + "/sprite"
      + "?camera=" + encodeURIComponent(camera)
      + "&frame=" + encodeURIComponent(frame)
      + "&kind=" + encodeURIComponent(kind);
  }

  function preloadSprite(url, key) {
    if (state.spritePreloads.has(key)) return state.spritePreloads.get(key);
    var promise = new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () { resolve(url); };
      image.onerror = function () { reject(new Error("Tactile sprite failed to load")); };
      image.src = url;
    });
    state.spritePreloads.set(key, promise);
    if (state.spritePreloads.size > 40) {
      var first = state.spritePreloads.keys().next();
      if (!first.done) state.spritePreloads.delete(first.value);
    }
    return promise;
  }

  function prefetchFollowingSprites(record, camera, syncIndex, kind) {
    var rows = state.series && Array.isArray(state.series.sync_frames) ? state.series.sync_frames : [];
    for (var offset = 1; offset <= 3; offset += 1) {
      var row = rows[syncIndex + offset];
      if (!row) break;
      var frame = Number(row.frame);
      var key = record.id + "|" + camera + "|" + frame + "|" + kind;
      preloadSprite(spriteUrl(record, camera, frame, kind), key).catch(function () {});
    }
  }

  function applySprite(record, camera, matchedFrame, syncIndex, kind) {
    var key = record.id + "|" + camera + "|" + matchedFrame + "|" + kind;
    if (state.appliedSpriteKey === key || state.requestedSpriteKey === key) {
      prefetchFollowingSprites(record, camera, syncIndex, kind);
      return;
    }
    state.requestedSpriteKey = key;
    var serial = ++state.spriteSerial;
    var url = spriteUrl(record, camera, matchedFrame, kind);
    prefetchFollowingSprites(record, camera, syncIndex, kind);
    preloadSprite(url, key).then(function () {
      if (serial !== state.spriteSerial || state.requestedSpriteKey !== key) return;
      grid.style.setProperty("--tactile-sprite-url", 'url("' + url.replace(/"/g, "%22") + '")');
      state.appliedSpriteKey = key;
    }).catch(function () {
      if (serial !== state.spriteSerial) return;
      statusNode.textContent += " · tactile image unavailable";
    });
  }

  function formatAxisNumber(value) {
    if (!Number.isFinite(value)) return "—";
    var magnitude = Math.abs(value);
    if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) return value.toExponential(2);
    return value.toFixed(magnitude >= 10 ? 2 : 3);
  }

  function renderCurve() {
    if (!state.series || !state.series.fingers) {
      curveNode.innerHTML = '<div class="placeholder">No f6 history.</div>';
      return;
    }
    var finger = String(byId("curveFingerSelect").value || "index");
    var points = Array.isArray(state.series.fingers[finger]) ? state.series.fingers[finger] : [];
    if (!points.length) {
      curveNode.innerHTML = '<div class="placeholder">No f6 history for ' + escapeHtml(LABELS[finger] || finger) + '.</div>';
      return;
    }
    var frameMin = Number(state.series.frame_min);
    var frameMax = Number(state.series.frame_max);
    if (!Number.isFinite(frameMin)) frameMin = Number(points[0].frame) || 0;
    if (!Number.isFinite(frameMax)) frameMax = Number(points[points.length - 1].frame) || frameMin + 1;
    if (frameMax <= frameMin) frameMax = frameMin + 1;

    var values = [];
    points.forEach(function (point) {
      (Array.isArray(point.f6) ? point.f6.slice(0, 6) : []).forEach(function (value) {
        var number = Number(value);
        if (Number.isFinite(number)) values.push(number);
      });
    });
    if (!values.length) {
      curveNode.innerHTML = '<div class="placeholder">No numeric f6 history.</div>';
      return;
    }

    var yMin = Math.min.apply(Math, values);
    var yMax = Math.max.apply(Math, values);
    if (yMax <= yMin) { yMin -= 0.5; yMax += 0.5; }
    var width = 640;
    var height = 150;
    var xFor = function (frame) {
      return Math.max(0, Math.min(width, (Number(frame) - frameMin) / (frameMax - frameMin) * width));
    };
    var yFor = function (value) {
      return Math.max(0, Math.min(height, height - (Number(value) - yMin) / (yMax - yMin) * height));
    };
    var traces = [0, 1, 2, 3, 4, 5].map(function (component) {
      var coords = points.map(function (point) {
        var value = Array.isArray(point.f6) ? Number(point.f6[component]) : NaN;
        if (!Number.isFinite(value)) return null;
        return xFor(point.frame).toFixed(2) + "," + yFor(value).toFixed(2);
      }).filter(Boolean);
      return coords.length
        ? '<polyline class="tactile-trace trace-' + component + '" points="' + coords.join(" ") + '"></polyline>'
        : "";
    }).join("");
    var playheadX = xFor(state.currentFrame);
    curveNode.innerHTML = '<div class="tactile-axis"><span>' + escapeHtml(formatAxisNumber(yMax))
      + '</span><span>' + escapeHtml(formatAxisNumber(yMin)) + '</span></div>'
      + '<svg class="tactile-curve-svg" viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none" role="img" aria-label="'
      + escapeHtml(LABELS[finger] || finger) + ' f6 history">'
      + '<line class="gridline" x1="0" y1="37.5" x2="640" y2="37.5"></line>'
      + '<line class="gridline" x1="0" y1="75" x2="640" y2="75"></line>'
      + '<line class="gridline" x1="0" y1="112.5" x2="640" y2="112.5"></line>'
      + traces
      + '<line class="playhead" data-playhead x1="' + playheadX.toFixed(2) + '" y1="0" x2="' + playheadX.toFixed(2) + '" y2="' + height + '"></line>'
      + '</svg>';
    curveNode.dataset.frameMin = String(frameMin);
    curveNode.dataset.frameMax = String(frameMax);
  }

  function updateCurvePlayhead(frame) {
    var playhead = curveNode.querySelector("[data-playhead]");
    if (!playhead) return;
    var frameMin = Number(curveNode.dataset.frameMin);
    var frameMax = Number(curveNode.dataset.frameMax);
    if (!Number.isFinite(frameMin) || !Number.isFinite(frameMax) || frameMax <= frameMin) return;
    var x = Math.max(0, Math.min(640, (frame - frameMin) / (frameMax - frameMin) * 640));
    playhead.setAttribute("x1", x.toFixed(2));
    playhead.setAttribute("x2", x.toFixed(2));
  }

  function ensureSeries(record, camera) {
    var key = record.id + "|" + camera;
    if (state.seriesKey === key && state.series) return Promise.resolve(state.series);
    if (state.loadingSeriesKey === key && state.loadingSeriesPromise) return state.loadingSeriesPromise;

    var serial = ++state.seriesSerial;
    statusNode.textContent = "Loading tactile timeline…";
    state.loadingSeriesKey = key;
    state.loadingSeriesPromise = (async function () {
      try {
        var response = await fetch(
          "/api/tactile/" + encodeURIComponent(record.id) + "/series?camera=" + encodeURIComponent(camera),
          { cache: "no-store" }
        );
        var body = await response.json();
        if (serial !== state.seriesSerial) return null;
        if (!response.ok) throw new Error(body.error || "Tactile timeline request failed");
        state.seriesKey = key;
        state.series = body.tactile;
        state.eventLookup = buildEventLookup(state.series);
        state.lastSyncKey = "";
        state.appliedSpriteKey = "";
        state.requestedSpriteKey = "";
        renderCurve();
        return state.series;
      } catch (error) {
        if (serial !== state.seriesSerial) return null;
        state.seriesKey = key;
        state.series = null;
        state.eventLookup = null;
        statusNode.textContent = "Tactile unavailable: " + String(error.message || error);
        curveNode.innerHTML = '<div class="placeholder">f6 history unavailable.</div>';
        grid.innerHTML = "";
        return null;
      } finally {
        if (state.loadingSeriesKey === key) {
          state.loadingSeriesKey = "";
          state.loadingSeriesPromise = null;
        }
      }
    })();
    return state.loadingSeriesPromise;
  }

  function renderSynchronizedFrame(record, camera, match, frame, force) {
    if (!match || !match.row) {
      statusNode.textContent = "No synchronized tactile frame.";
      grid.innerHTML = "";
      return;
    }
    var syncRow = match.row;
    var kind = String(byId("kindSelect").value || "deform");
    var syncKey = record.id + "|" + camera + "|" + syncRow.sync_row + "|" + kind;
    if (!force && syncKey === state.lastSyncKey) {
      updateCurvePlayhead(frame);
      return;
    }
    state.lastSyncKey = syncKey;
    grid.style.setProperty("--tactile-cell-aspect", kind === "raw" ? "4 / 3" : "1 / 1");
    var completeText = syncRow.complete === false ? "incomplete" : "complete";
    statusNode.textContent = camera + " frame " + frame
      + " → sync frame " + syncRow.frame
      + " · row " + syncRow.sync_row + " · " + completeText;
    grid.innerHTML = FINGERS.map(function (finger) {
      return renderFinger(finger, currentFingerData(syncRow, finger), kind);
    }).join("");
    state.appliedSpriteKey = "";
    applySprite(record, camera, Number(syncRow.frame), match.index, kind);
    updateCurvePlayhead(frame);
  }

  async function refreshTactile(frame, force) {
    var record = selectedRecord();
    if (!record) return;
    updateReadout(frame);
    var series = await ensureSeries(record, state.camera);
    if (!series) return;
    renderSynchronizedFrame(record, state.camera, nearestSyncFrame(state.currentFrame), state.currentFrame, Boolean(force));
  }

  function installVideoDrivenRefresh() {
    if (typeof video.requestVideoFrameCallback === "function") {
      var onVideoFrame = function (_now, metadata) {
        var record = selectedRecord();
        if (record) refreshTactile(frameFromVideo(record, metadata && metadata.mediaTime), false);
        state.videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
      };
      state.videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
      return;
    }
    var onAnimationFrame = function () {
      var record = selectedRecord();
      if (record && !video.paused && !video.ended) refreshTactile(frameFromVideo(record, video.currentTime), false);
      state.animationFrameId = window.requestAnimationFrame(onAnimationFrame);
    };
    state.animationFrameId = window.requestAnimationFrame(onAnimationFrame);
  }

  async function loadRollouts() {
    byId("datasetStatus").textContent = "Loading tactile rollouts…";
    try {
      var response = await fetch("/api/rollouts", { cache: "no-store" });
      var body = await response.json();
      if (!response.ok) throw new Error(body.error || "Could not load rollouts");
      state.rollouts = (body.rollouts || []).filter(supportsTactile);
      byId("datasetStatus").textContent = state.rollouts.length + " tactile rollouts";
      renderRolloutList();
      if (state.filtered.length) selectRollout(state.filtered[0].id);
      else {
        byId("emptyState").innerHTML = '<div><p class="eyebrow">NO TACTILE DATA</p><h2>No loaded rollout exposes synchronized tactile paths.</h2></div>';
      }
    } catch (error) {
      byId("datasetStatus").textContent = "Dataset unavailable";
      byId("emptyState").innerHTML = '<div><p class="eyebrow">LOAD ERROR</p><h2>' + escapeHtml(error.message || error) + '</h2></div>';
    }
  }

  byId("rolloutList").addEventListener("click", function (event) {
    var button = event.target && event.target.closest ? event.target.closest("[data-rollout-id]") : null;
    if (button) selectRollout(button.dataset.rolloutId);
  });
  byId("searchInput").addEventListener("input", renderRolloutList);
  byId("previousButton").addEventListener("click", function () { navigate(-1); });
  byId("nextButton").addEventListener("click", function () { navigate(1); });
  byId("cameraSelect").addEventListener("change", async function () {
    var record = selectedRecord();
    if (!record) return;
    state.camera = String(this.value || "cam_high");
    state.seriesKey = "";
    state.series = null;
    state.loadingSeriesKey = "";
    state.loadingSeriesPromise = null;
    state.seriesSerial += 1;
    state.eventLookup = null;
    state.lastSyncKey = "";
    state.appliedSpriteKey = "";
    state.requestedSpriteKey = "";
    setVideoSource(record, true);
    await ensureSeries(record, state.camera);
    refreshTactile(state.currentFrame, true);
  });
  byId("kindSelect").addEventListener("change", function () {
    state.lastSyncKey = "";
    state.appliedSpriteKey = "";
    state.requestedSpriteKey = "";
    state.spriteSerial += 1;
    refreshTactile(state.currentFrame, true);
  });
  byId("curveFingerSelect").addEventListener("change", renderCurve);
  byId("playButton").addEventListener("click", function () {
    if (video.paused || video.ended) {
      var promise = video.play();
      if (promise && typeof promise.catch === "function") promise.catch(function () {});
    } else {
      video.pause();
    }
  });
  byId("stepBackButton").addEventListener("click", function () { seekFrame(state.currentFrame - 1); });
  byId("stepForwardButton").addEventListener("click", function () { seekFrame(state.currentFrame + 1); });
  frameSlider.addEventListener("input", function () { seekFrame(Number(this.value)); });
  video.addEventListener("play", function () { byId("playButton").textContent = "Pause"; });
  video.addEventListener("pause", function () {
    byId("playButton").textContent = "Play";
    refreshTactile(frameFromVideo(selectedRecord(), video.currentTime), true);
  });
  video.addEventListener("seeked", function () { refreshTactile(frameFromVideo(selectedRecord(), video.currentTime), true); });
  video.addEventListener("ended", function () { byId("playButton").textContent = "Play"; });
  video.addEventListener("timeupdate", function () { updateReadout(frameFromVideo(selectedRecord(), video.currentTime)); });
  document.addEventListener("keydown", function (event) {
    var target = event.target;
    if (target && (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA")) return;
    if (event.key === " ") {
      event.preventDefault();
      byId("playButton").click();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      seekFrame(state.currentFrame - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      seekFrame(state.currentFrame + 1);
    } else if (event.key === ",") {
      navigate(-1);
    } else if (event.key === ".") {
      navigate(1);
    }
  });

  installVideoDrivenRefresh();
  loadRollouts();
})();
