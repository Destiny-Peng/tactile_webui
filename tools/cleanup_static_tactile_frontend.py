from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding='utf-8')
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one match, found {count}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')


app = ROOT / 'webui/static/tactile/app.js'
replace_once(
    app,
    '''    lastSyncKey: "",\n    appliedSpriteKey: "",\n    requestedSpriteKey: "",\n    spriteSerial: 0,\n    spritePreloads: new Map(),\n    videoFrameCallbackId: 0,''',
    '''    lastSyncKey: "",\n    videoFrameCallbackId: 0,''',
    'remove sprite state',
)
replace_once(
    app,
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    state.appliedSpriteKey = "";\n    state.requestedSpriteKey = "";\n    state.spriteSerial += 1;\n    grid.innerHTML = "";''',
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    grid.innerHTML = "";''',
    'remove sprite reset',
)
start = app.read_text(encoding='utf-8').index('  function spriteUrl(')
end = app.read_text(encoding='utf-8').index('  function formatAxisNumber(', start)
text = app.read_text(encoding='utf-8')
app.write_text(text[:start] + text[end:], encoding='utf-8')
replace_once(
    app,
    '''    grid.innerHTML = FINGERS.map(function (finger) {\n      return renderFinger(record, finger, currentFingerData(syncRow, finger), kind);\n    }).join("");\n    state.appliedSpriteKey = "";\n    updateCurvePlayhead(frame);''',
    '''    grid.innerHTML = FINGERS.map(function (finger) {\n      return renderFinger(record, finger, currentFingerData(syncRow, finger), kind);\n    }).join("");\n    updateCurvePlayhead(frame);''',
    'remove applied sprite reset',
)
replace_once(
    app,
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    state.appliedSpriteKey = "";\n    state.requestedSpriteKey = "";\n    setVideoSource(record, true);''',
    '''    state.eventLookup = null;\n    state.lastSyncKey = "";\n    setVideoSource(record, true);''',
    'camera sprite reset',
)
replace_once(
    app,
    '''  byId("kindSelect").addEventListener("change", function () {\n    state.lastSyncKey = "";\n    state.appliedSpriteKey = "";\n    state.requestedSpriteKey = "";\n    state.spriteSerial += 1;\n    refreshTactile(state.currentFrame, true);\n  });''',
    '''  byId("kindSelect").addEventListener("change", function () {\n    state.lastSyncKey = "";\n    refreshTactile(state.currentFrame, true);\n  });''',
    'kind sprite reset',
)

styles = ROOT / 'webui/static/tactile/styles.css'
replace_once(
    styles,
    '''.tactile-sprite {\n  background-image: var(--tactile-sprite-url);\n  background-repeat: no-repeat;\n  background-size: 500% 100%;\n}\n.tactile-sprite.finger-0 { background-position: 0% 50%; }\n.tactile-sprite.finger-1 { background-position: 25% 50%; }\n.tactile-sprite.finger-2 { background-position: 50% 50%; }\n.tactile-sprite.finger-3 { background-position: 75% 50%; }\n.tactile-sprite.finger-4 { background-position: 100% 50%; }\n''',
    '',
    'remove sprite css',
)
