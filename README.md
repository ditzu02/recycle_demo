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
- metal-only contamination evaluation inside that gate
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

Point the runtime at a different brain host:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --brain-base-url http://127.0.0.1:8000
```

Useful runtime knobs:

- `EDGE_DEVICE_ID`
- `EDGE_BRAIN_BASE_URL`
- `EDGE_EVENT_ENDPOINT_URL`
- `EDGE_HEARTBEAT_ENDPOINT_URL`
- `EDGE_CAMERA_INDEX`
- `EDGE_CAMERA_WIDTH`
- `EDGE_CAMERA_HEIGHT`
- `EDGE_CONFIDENCE_THRESHOLD`
- `EDGE_STABLE_AFTER_FRAMES`
- `EDGE_MAX_MISSED_FRAMES`
- `EDGE_MIN_IN_ZONE_FRAMES_FOR_EVALUATION`
- `EDGE_EVALUATION_ZONE`
- `EDGE_SHOW_PREVIEW`

### Edge Notes

- The current demo runtime tracks the whole frame, but only evaluates objects after they become stable and dwell inside the evaluation zone.
- The current demo runtime filters to `Metal` by default because the contamination CNN is metal-specific.
- The default evaluation zone is a normalized lower-frame rectangle: `0.20,0.55,0.80,0.95`.
- `EDGE_INSPECTION_ZONE` is still accepted as a compatibility alias for `EDGE_EVALUATION_ZONE`.
- The runtime emits one finalized Brain-v1 event per completed tracked-object lifecycle.
- A track is evaluated once per pass through the gate and emits once when it leaves the zone or disappears after evaluation.
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

### API

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

### Demo Notes

- The Overview and Events pages auto-refresh every 5 seconds during a live demo.
- Device last-seen ordering is based on when the brain received the event or heartbeat.
- Event and object tables show both the device-reported timestamp and the brain receive time for provenance.
- Standalone mock demo mode uses the default seeding behavior or `BRAIN_SEED_MOCK=1`.
- Real-edge mode should use `BRAIN_SEED_MOCK=0` and avoid running `python -m brain.mock.simulator`.
