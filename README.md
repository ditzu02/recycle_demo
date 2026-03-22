# Recycle Demo

This repository now contains two local proof-of-concept parts:

- `demo_webcam.py`: the existing edge-side YOLO + CNN webcam demo
- `brain/`: a central coordination prototype that receives, stores, and visualizes inference events from future Raspberry Pi devices

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

Start the brain server:

```bash
./.venv/bin/pip install -r requirements-brain.txt
./.venv/bin/python -m brain.backend.app
```

Open the dashboard:

- `http://127.0.0.1:8000/`

The server seeds deterministic mock events into SQLite when the database is empty, so the dashboard is populated on first launch.

### Simulate Future Raspberry Pi Devices

With the server running, send mock events:

```bash
./.venv/bin/python -m brain.mock.simulator --count 12 --devices 3
```

This posts seeded JSON payloads to:

- `POST http://127.0.0.1:8000/api/inference`

### API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Example inference payload:

```json
{
  "device_id": "pi_01",
  "timestamp": "2026-01-01T12:00:00",
  "objects": [
    {
      "label": "Metal",
      "confidence": 0.91,
      "score": 87,
      "decision": "Accept",
      "bbox": [100, 120, 220, 260],
      "dirty_probability": 0.12
    }
  ]
}
```

### Future Integration Notes

- `POST /api/inference` already accepts the simple mock payload above.
- The schema layer also accepts a richer future payload shape based on the current edge-node analysis, including `detections`, frame metadata, and contamination refinement fields.
- Real Raspberry Pi devices can later replace the mock simulator without changing the persistence or dashboard layers.
