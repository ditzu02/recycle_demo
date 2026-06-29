# Recycle Demo

This repository now contains three local proof-of-concept parts:

- `demo_webcam.py`: the existing edge-side YOLO + CNN webcam demo
- `edge/`: the initial real edge runtime package for local demos and future Raspberry Pi porting
- `brain/`: a central coordination prototype that receives, stores, and visualizes finalized inspection events and heartbeats from future Raspberry Pi devices

## Edge Runtime

`edge/` is the maintainable runtime path that replaces the one-file webcam demo for real device flow work.

Current v1 scope:

- camera capture via OpenCV
- YOLOv8 bbox detections
- full-frame tracking with a designated evaluation zone gate
- multi-class detection with metal-only contamination evaluation inside that gate
- lightweight per-object temporal tracking and stabilization
- explicit `Accept` / `Review` / `Reject` decisions
- canonical Brain-v1 event and heartbeat payload mapping
- HTTP transport with bounded retries and in-memory resend state

### Edge Layout

```text
edge/
  __main__.py
  camera.py
  config.py
  contamination.py
  decision.py
  detection.py
  filtering.py
  payloads.py
  runtime.py
  stabilization.py
  tracking.py
  transport.py
  types.py
```

### Run Locally

The edge runtime reuses the existing local demo dependencies already present in the virtual environment, along with:

- `best8S.pt`
- `metal_contamination_cnn_best.pt`

Start the edge runtime in headless mode:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01
```

Show the local preview window for desktop demos:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --show
```

Save local debug images for each finalized event:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --show --save-debug-images
```

Save a run log with FPS, latency, event attempts, duplicate responses, conflicts, and heartbeat counts:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --show --save-run-log
```

Point the runtime at a different brain host:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --brain-base-url http://127.0.0.1:8000
```

Export the current YOLO weights to ONNX for Raspberry Pi performance testing:

```bash
./.venv/bin/python -c "from ultralytics import YOLO; YOLO('best8S.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
```

Run with the exported ONNX detector:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --yolo-model-path best8S.onnx
```

Reviewer-data run profile using the ONNX detector at 416 px:

```bash
./.venv/bin/python -m edge \
  --device-id edge_demo_01 \
  --brain-base-url http://<laptop-ip>:8000 \
  --yolo-model-path best8S.onnx \
  --yolo-imgsz 416 \
  --show \
  --save-debug-images \
  --save-run-log
```

Useful runtime knobs:

- `EDGE_DEVICE_ID`
- `EDGE_BRAIN_BASE_URL`
- `EDGE_EVENT_ENDPOINT_URL`
- `EDGE_HEARTBEAT_ENDPOINT_URL`
- `EDGE_CAMERA_INDEX`
- `EDGE_CAMERA_WIDTH`
- `EDGE_CAMERA_HEIGHT`
- `EDGE_YOLO_MODEL_PATH`
- `EDGE_YOLO_IMGSZ`
- `EDGE_YOLO_CONFIDENCE_FLOOR`
- `EDGE_YOLO_CLASS_NAMES`
- `EDGE_CONFIDENCE_THRESHOLD`
- `EDGE_STABLE_AFTER_FRAMES`
- `EDGE_MAX_MISSED_FRAMES`
- `EDGE_MIN_IN_ZONE_FRAMES_FOR_EVALUATION`
- `EDGE_CONTAMINATION_SAMPLE_COUNT`
- `EDGE_LABEL_ACCEPT_CONFIDENCE`
- `EDGE_ALLOWED_CLASSES`
- `EDGE_EVALUATION_ZONE`
- `EDGE_DEBUG_SAVE_IMAGES`
- `EDGE_DEBUG_OUTPUT_DIR`
- `EDGE_SAVE_RUN_LOG`
- `EDGE_RUN_LOG_PATH`
- `EDGE_SHOW_PREVIEW`

### Edge Notes

- The current demo runtime tracks the whole frame, but only evaluates objects after they become stable and dwell inside the evaluation zone.
- The runtime still defaults to `Metal` for backward compatibility, but `EDGE_ALLOWED_CLASSES` can now be widened to supported non-metal labels such as `Plastic` or `Glass`.
- YOLO model loading supports `.pt` and exported `.onnx` files through `EDGE_YOLO_MODEL_PATH` / `--yolo-model-path`.
- Exported ONNX models usually carry class names, but `EDGE_YOLO_CLASS_NAMES` can provide a comma-separated fallback if metadata is missing.
- The contamination CNN is only used for tracks whose final label is `Metal`.
- Metal contamination decisions average up to `3` eligible in-zone CNN samples by default.
- Supported non-metal classes are decided from stability plus label confidence only; they do not claim CNN-based contamination refinement.
- The default high-confidence accept threshold for label-driven decisions is `0.85`.
- The default evaluation zone is a normalized lower-frame rectangle: `0.20,0.55,0.80,0.95`.
- `EDGE_INSPECTION_ZONE` is still accepted as a compatibility alias for `EDGE_EVALUATION_ZONE`.
- The runtime emits one finalized Brain-v1 event per completed tracked-object lifecycle.
- A track is evaluated once per pass through the gate and emits once when it leaves the zone or disappears after evaluation.
- When debug image saving is enabled, the runtime stores the finalized event's evaluation crop and an annotated frame in `edge_debug/` by default.
- When run logging is enabled, the runtime stores a JSON summary in `edge_debug/` by default with frame timing, FPS, event HTTP attempts, duplicate responses, conflict responses, heartbeat counts, and latency summaries.
- `event_id` includes the device id, a runtime session token, the finalized frame index, and the track number so retries stay stable without reusing ids across separate runs.
- `inspection_outcome` is always included as an object and stays `{}` in this phase.
- Heartbeats are sent to `POST /api/heartbeat`.
- Retries are in-memory only; there is no disk-backed offline queue yet.

## Central Brain Prototype

The central brain is intentionally lightweight:

- backend: Python standard library WSGI server
- storage: SQLite
- UI: server-rendered Jinja2 templates
- transport: JSON over HTTP
- mock data: seeded startup data plus a simulator script

This stack was chosen because the current local environment does not yet have `fastapi` or `flask` installed. The code is organized so the HTTP layer can later be swapped to FastAPI without rewriting the schema, database, or mock generator.

### Project Layout

```text
brain/
  backend/
    app.py
  database/
    repository.py
  models/
    schema.py
  mock/
    seed.py
    simulator.py
  static/
    styles.css
  templates/
    base.html
    overview.html
    events.html
```

### Run Locally

Start the brain server with the current default demo-compatible behavior:

```bash
./.venv/bin/pip install -r requirements-brain.txt
./.venv/bin/python -m brain.backend.app
```

Explicit mock-enabled standalone demo mode:

```bash
BRAIN_SEED_MOCK=1 ./.venv/bin/python -m brain.backend.app
```

Real-edge mode with startup mock seeding disabled:

```bash
BRAIN_SEED_MOCK=0 ./.venv/bin/python -m brain.backend.app
```

Bind the brain to the LAN for a Raspberry Pi integration demo:

```bash
BRAIN_HOST=0.0.0.0 BRAIN_PORT=8000 BRAIN_SEED_MOCK=0 ./.venv/bin/python -m brain.backend.app
```

Open the dashboard:

- `http://127.0.0.1:8000/`

If `BRAIN_SEED_MOCK` is unset, the server keeps the current compatible behavior and seeds deterministic mock events when the database is empty.
If `BRAIN_SEED_MOCK=0`, startup still initializes the database but skips mock event insertion.

### Simulate Future Raspberry Pi Devices

With the server running, send mock events:

```bash
./.venv/bin/python -m brain.mock.simulator --count 12 --devices 3
```

This posts seeded Brain-v1 finalized inspection events to:

- `POST http://127.0.0.1:8000/api/inference`

### Brain Interface

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Canonical Brain-v1 finalized inspection event:

```json
{
  "schema_version": "brain-v1",
  "event_type": "inspection.finalized",
  "event_id": "pi_01-final-0001",
  "device_id": "pi_01",
  "timestamp": "2026-01-01T12:00:00Z",
  "source": {
    "type": "raspberry_pi_5",
    "index": 1
  },
  "frame": {
    "width": 1280,
    "height": 720,
    "frame_index": 42
  },
  "inspection_outcome": {},
  "objects": [
    {
      "object_id": "obj-0001",
      "class_id": 2,
      "label": "Metal",
      "confidence": 0.91,
      "bbox": {
        "x1": 100,
        "y1": 120,
        "x2": 220,
        "y2": 260
      },
      "score": 87,
      "decision": "Accept",
      "contamination_status": "CLEAN",
      "dirty_probability": 0.12,
      "clean_probability": 0.88,
      "refinement": {
        "applied": true,
        "probabilities": {
          "dirty": 0.12,
          "clean": 0.88
        }
      }
    }
  ]
}
```

Canonical Brain-v1 heartbeat:

```json
{
  "schema_version": "brain-v1",
  "device_id": "pi_01",
  "timestamp": "2026-01-01T12:00:05Z",
  "status": "online"
}
```

### Brain-v1 behavior

- `POST /api/inference` expects one finalized inspection result event per request.
- `schema_version` must be `brain-v1`.
- `event_type` must be `inspection.finalized`.
- `event_id` is the retry-stable ingest key and must be globally unique across all devices.
- `201 Created` means the event was accepted and stored.
- `200 OK` with `result: "duplicate"` means the same `event_id` was retried and no second event row was written.
- `409 Conflict` is returned only if an existing `event_id` is reused by a different `device_id`.
- `POST /api/heartbeat` updates device liveness without creating inspection rows.
- Heartbeat freshness and dashboard last-seen values are computed from the brain's server-side receive time, not the device clock.

### Legacy Compatibility

- The parser still accepts the older mock/simple payload shape where `schema_version`, `event_type`, and `inspection_outcome` are missing.
- The parser still accepts `detections` as an alias for `objects`.
- The parser still accepts `bbox` as either `[x1, y1, x2, y2]` or an object with `x1/y1/x2/y2`.

### Runtime Configuration

- `BRAIN_HOST`: bind address for the WSGI server. Default: `127.0.0.1`
- `BRAIN_PORT`: bind port for the WSGI server. Default: `8000`
- `BRAIN_SEED_MOCK`: startup mock seeding toggle. Default when unset: enabled. Truthy values: `1`, `true`, `yes`, `on`. Falsy values: `0`, `false`, `no`, `off`.
- `BRAIN_HEARTBEAT_FRESH_SECONDS`: heartbeat age treated as fresh. Default: `30`.
- `BRAIN_HEARTBEAT_OFFLINE_SECONDS`: heartbeat age treated as offline. Default: `90`.

### Demo Notes

- The Overview live inference stream polls compact JSON every 3 seconds; the Events page keeps a 5-second refresh during live demos.
- Device state is derived from heartbeat receive time: unknown before first heartbeat, online while fresh, stale after the fresh threshold, and offline after the offline threshold or an explicit offline heartbeat.
- Device last-seen ordering is based on when the brain received the event or heartbeat.
- Object result tables show both the device-reported timestamp and the brain receive time for provenance.
- Standalone mock demo mode uses the default seeding behavior or `BRAIN_SEED_MOCK=1`.
- Real-edge mode should use `BRAIN_SEED_MOCK=0` and avoid running `python -m brain.mock.simulator`.
