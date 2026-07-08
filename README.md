# Recycle Demo

[![Tests](https://github.com/ditzu02/recycle_demo/actions/workflows/tests.yml/badge.svg)](https://github.com/ditzu02/recycle_demo/actions/workflows/tests.yml)

Recycle Demo is a local waste-inspection demo. It has an edge runtime that reads camera frames and a small dashboard that stores and shows inspection events.

Main parts:

- `demo_webcam.py`: original webcam demo.
- `edge/`: camera, detection, tracking, contamination checks, and HTTP upload.
- `brain/`: local server, SQLite storage, mock data, and dashboard pages.

## Repository

Source repository: [https://github.com/ditzu02/recycle_demo](https://github.com/ditzu02/recycle_demo)

Visibility: public.

The repository is meant to contain the project source without compiled application binaries. Local databases, debug images, logs, caches, and exported model files are runtime output and are not needed for the source-code submission.

## Build

There is no compiled application build. After installing the dependencies, check the source with:

```bash
./.venv/bin/python -m compileall -q brain edge tests demo_webcam.py
./.venv/bin/python -m unittest discover -s tests -p 'test*.py'
```

Optional ONNX export for Raspberry Pi testing:

```bash
./.venv/bin/python -c "from ultralytics import YOLO; YOLO('best8S.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
```

## Installation

Create a virtual environment and install the dashboard/test dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-brain.txt
./.venv/bin/pip install -r requirements-test.txt
```

The edge runtime also needs the packages used by the webcam demo:

```bash
./.venv/bin/pip install numpy opencv-python torch torchvision ultralytics
```

Required model files for edge runs:

- `best8S.pt`
- `metal_contamination_cnn_best.pt`

## Launch

Start the dashboard:

```bash
./.venv/bin/python -m brain.backend.app
```

Open `http://127.0.0.1:8000/`.

Start the edge runtime in another terminal:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --brain-base-url http://127.0.0.1:8000
```

For a desktop preview window:

```bash
./.venv/bin/python -m edge --device-id edge_demo_01 --show
```

For a real edge run without startup mock data:

```bash
BRAIN_SEED_MOCK=0 ./.venv/bin/python -m brain.backend.app
```

To expose the dashboard on the LAN:

```bash
BRAIN_HOST=0.0.0.0 BRAIN_PORT=8000 BRAIN_SEED_MOCK=0 ./.venv/bin/python -m brain.backend.app
```

## Project layout

```text
brain/
  backend/app.py          WSGI server and routes
  database/repository.py  SQLite access
  models/schema.py        request parsing and data models
  mock/                   seed data and event simulator
  templates/              dashboard pages
  static/                 CSS and browser assets

edge/
  __main__.py             command-line entry point
  camera.py               OpenCV capture
  detection.py            YOLO wrapper
  contamination.py        CNN contamination check
  runtime.py              main edge loop
  transport.py            HTTP upload and retry logic
```

## Edge runtime

The edge process captures frames, runs YOLO detections, tracks objects through an evaluation zone, checks metal objects with the contamination CNN, and sends final events to the dashboard.

Useful commands:

```bash
# Save debug crops and annotated frames
./.venv/bin/python -m edge --device-id edge_demo_01 --show --save-debug-images

# Save run metrics under edge_debug/
./.venv/bin/python -m edge --device-id edge_demo_01 --show --save-run-log

# Use an exported ONNX detector
./.venv/bin/python -m edge --device-id edge_demo_01 --yolo-model-path best8S.onnx
```

Common edge settings:

- `EDGE_DEVICE_ID`
- `EDGE_BRAIN_BASE_URL`
- `EDGE_CAMERA_INDEX`
- `EDGE_YOLO_MODEL_PATH`
- `EDGE_YOLO_IMGSZ`
- `EDGE_ALLOWED_CLASSES`
- `EDGE_EVALUATION_ZONE`
- `EDGE_SHOW_PREVIEW`
- `EDGE_DEBUG_SAVE_IMAGES`
- `EDGE_SAVE_RUN_LOG`

## Brain dashboard

The brain server uses Python's standard WSGI server, SQLite, and Jinja2 templates. It receives inspection events and heartbeats over HTTP.

Run with demo seed data:

```bash
BRAIN_SEED_MOCK=1 ./.venv/bin/python -m brain.backend.app
```

Send mock events while the server is running:

```bash
./.venv/bin/python -m brain.mock.simulator --count 12 --devices 3
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

HTTP endpoints:

- `POST /api/inference`: stores one `brain-v1` finalized inspection event.
- `POST /api/heartbeat`: updates device liveness.
- `GET /health`: returns server health.

`event_id` is used for retry-safe ingestion. Reposting the same event returns a duplicate result instead of inserting a second row. Reusing an existing `event_id` from another device returns a conflict.

Brain settings:

- `BRAIN_HOST`: bind address. Default: `127.0.0.1`.
- `BRAIN_PORT`: bind port. Default: `8000`.
- `BRAIN_SEED_MOCK`: seed demo rows on startup. Use `0` for real edge runs.
- `BRAIN_HEARTBEAT_FRESH_SECONDS`: heartbeat age treated as fresh. Default: `30`.
- `BRAIN_HEARTBEAT_OFFLINE_SECONDS`: heartbeat age treated as offline. Default: `90`.
