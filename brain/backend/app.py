from __future__ import annotations

import json
import logging
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from jinja2 import Environment, FileSystemLoader, select_autoescape

from brain.database.repository import BrainRepository, EventConflictError
from brain.mock.seed import seed_repository_if_empty
from brain.models.schema import (
    CANONICAL_EVENT_TYPE,
    CANONICAL_SCHEMA_VERSION,
    SchemaValidationError,
    event_to_dict,
    heartbeat_to_dict,
    parse_heartbeat_payload,
    parse_inference_payload,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "brain.db"
DASHBOARD_REFRESH_SECONDS = 5
LOGGER = logging.getLogger(__name__)


class BrainApplication:
    def __init__(self, repository: BrainRepository):
        self.repository = repository
        self.templates = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        self.templates.filters["datetime_human"] = _format_timestamp_human

    def __call__(self, environ: dict, start_response: Callable):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")
        query = parse_qs(environ.get("QUERY_STRING", ""))

        try:
            if method == "GET" and path == "/":
                return self._overview_page(start_response)
            if method == "GET" and path == "/events":
                return self._events_page(start_response, query)
            if method == "GET" and path == "/api":
                return self._api_page(start_response)
            if method == "GET" and path == "/health":
                return self._json(start_response, 200, {"status": "ok"})
            if method == "GET" and path == "/api/overview":
                return self._json(start_response, 200, self.repository.get_overview())
            if method == "GET" and path == "/api/events":
                limit = _int_from_query(query, "limit", default=20, maximum=100)
                payload = {
                    "events": self.repository.get_recent_events(limit=limit),
                    "objects": self.repository.get_recent_objects_for_event_page(
                        event_limit=limit,
                        event_offset=0,
                        object_limit=limit,
                    ),
                }
                return self._json(start_response, 200, payload)
            if method == "POST" and path == "/api/inference":
                return self._post_inference(start_response, environ)
            if method == "POST" and path == "/api/heartbeat":
                return self._post_heartbeat(start_response, environ)
            if method == "GET" and path.startswith("/static/"):
                return self._static_file(start_response, path)
        except SchemaValidationError as exc:
            LOGGER.warning(
                "validation failure path=%s remote=%s detail=%s",
                path,
                environ.get("REMOTE_ADDR", "-"),
                exc,
            )
            return self._json(start_response, 400, {"status": "error", "detail": str(exc)})
        except EventConflictError as exc:
            LOGGER.warning(
                "event_id conflict path=%s remote=%s detail=%s",
                path,
                environ.get("REMOTE_ADDR", "-"),
                exc,
            )
            return self._json(start_response, 409, {"status": "error", "detail": str(exc)})
        except FileNotFoundError:
            return self._json(start_response, 404, {"status": "error", "detail": "Not found"})
        except Exception as exc:  # pragma: no cover - demo-level fallback
            LOGGER.exception("unhandled application error path=%s", path)
            return self._json(start_response, 500, {"status": "error", "detail": str(exc)})

        return self._json(start_response, 404, {"status": "error", "detail": "Not found"})

    def _overview_page(self, start_response: Callable):
        overview = self.repository.get_overview(device_limit=8, recent_device_limit=8)
        template = self.templates.get_template("overview.html")
        body = template.render(
            title="Recycle Brain | Overview",
            page_name="Overview",
            page_description="Central monitoring dashboard for finalized inspection traffic, device liveness, and recent edge-node output.",
            active_page="overview",
            overview=overview,
            recent_objects=self.repository.get_recent_objects_for_event_page(
                event_limit=8,
                event_offset=0,
                object_limit=8,
            ),
        )
        return self._html(start_response, body)

    def _events_page(self, start_response: Callable, query: dict[str, list[str]]):
        page = _int_from_query(query, "page", default=1)
        limit = _int_from_query(query, "limit", default=25, maximum=100)
        offset = (page - 1) * limit
        events = self.repository.get_recent_events(limit=limit + 1, offset=offset)
        has_next = len(events) > limit
        events = events[:limit]
        template = self.templates.get_template("events.html")
        body = template.render(
            title="Recycle Brain | Events",
            page_name="Events",
            page_description="Recent finalized inspection events and object-level decisions reported by the distributed edge-AI nodes.",
            active_page="events",
            events=events,
            objects=self.repository.get_recent_objects_for_event_page(
                event_limit=limit,
                event_offset=offset,
            ),
            page=page,
            limit=limit,
            has_previous=page > 1,
            has_next=has_next,
            previous_page=page - 1,
            next_page=page + 1,
        )
        return self._html(start_response, body)

    def _api_page(self, start_response: Callable):
        overview = self.repository.get_overview(device_limit=8, recent_device_limit=8)
        events_sample_limit = 6
        events_payload = {
            "events": self.repository.get_recent_events(limit=events_sample_limit),
            "objects": self.repository.get_recent_objects_for_event_page(
                event_limit=events_sample_limit,
                event_offset=0,
                object_limit=events_sample_limit,
            ),
        }
        sample_inference_payload = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "event_type": CANONICAL_EVENT_TYPE,
            "event_id": "pi_01-final-0001",
            "device_id": "pi_01",
            "timestamp": "2026-01-01T12:00:00Z",
            "source": {
                "type": "raspberry_pi_5",
                "index": 1,
            },
            "frame": {
                "width": 1280,
                "height": 720,
                "frame_index": 42,
            },
            "inspection_outcome": {},
            "objects": [
                {
                    "object_id": "obj-0001",
                    "class_id": 2,
                    "label": "Metal",
                    "confidence": 0.91,
                    "bbox": {"x1": 100, "y1": 120, "x2": 220, "y2": 260},
                    "score": 87,
                    "decision": "Accept",
                    "contamination_status": "CLEAN",
                    "dirty_probability": 0.12,
                    "clean_probability": 0.88,
                    "refinement": {
                        "applied": True,
                        "probabilities": {
                            "dirty": 0.12,
                            "clean": 0.88,
                        },
                    },
                }
            ],
        }
        sample_heartbeat_payload = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "device_id": "pi_01",
            "timestamp": "2026-01-01T12:00:05Z",
            "status": "online",
        }
        template = self.templates.get_template("api.html")
        body = template.render(
            title="Recycle Brain | API",
            page_name="API",
            page_description="System endpoints, canonical Brain-v1 payload examples, and live response samples for the local central brain service.",
            active_page="api",
            system_summary={
                "status": "Local service available",
                "active_devices": overview["active_devices"],
                "total_events": overview["total_events"],
                "total_objects": overview["total_objects"],
            },
            endpoints=[
                {
                    "method": "GET",
                    "path": "/health",
                    "description": "Simple health-check endpoint for local service availability.",
                    "notes": "Returns a minimal status payload for connectivity checks.",
                    "sample_label": "Sample response",
                    "sample_body": _pretty_json({"status": "ok"}),
                },
                {
                    "method": "GET",
                    "path": "/api/overview",
                    "description": "Aggregated dashboard totals, per-device last-contact data, and heartbeat freshness summaries.",
                    "notes": "Used by the Overview dashboard page.",
                    "sample_label": "Live response sample",
                    "sample_body": _pretty_json(overview),
                },
                {
                    "method": "GET",
                    "path": "/api/events?limit=20",
                    "description": "Recent finalized inspection event rows and linked object-level results, ordered by brain receive time.",
                    "notes": f"Query parameter `limit` controls row count. Sample below uses {events_sample_limit}.",
                    "sample_label": "Live response sample",
                    "sample_body": _pretty_json(events_payload),
                },
                {
                    "method": "POST",
                    "path": "/api/inference",
                    "description": "Receives one finalized inspection result event per request and stores normalized event/object records.",
                    "notes": "Canonical Brain-v1 requests return `201 Created` on first ingest, `200 OK` with `result: duplicate` on retry, and `409` only for cross-device event_id conflicts.",
                    "sample_label": "Sample request body",
                    "sample_body": _pretty_json(sample_inference_payload),
                },
                {
                    "method": "POST",
                    "path": "/api/heartbeat",
                    "description": "Receives lightweight device liveness heartbeats and updates the last-seen device status view.",
                    "notes": "Heartbeat freshness is computed from brain-side receive time. A heartbeat newer than 30 seconds is treated as fresh.",
                    "sample_label": "Sample request body",
                    "sample_body": _pretty_json(sample_heartbeat_payload),
                },
            ],
            dashboard_refresh_seconds=DASHBOARD_REFRESH_SECONDS,
        )
        return self._html(start_response, body)

    def _post_inference(self, start_response: Callable, environ: dict):
        payload = self._read_json_body(environ)
        normalized = parse_inference_payload(payload)
        result = self.repository.store_event(normalized)
        remote_addr = environ.get("REMOTE_ADDR", "-")
        LOGGER.info(
            "%s finalized_event remote=%s device_id=%s event_id=%s objects=%s received_at=%s",
            result.result,
            remote_addr,
            normalized.device_id,
            normalized.event_uuid,
            len(normalized.objects),
            result.received_at,
        )
        return self._json(
            start_response,
            201 if result.result == "accepted" else 200,
            {
                "status": "ok",
                "result": result.result,
                "event_id": normalized.event_uuid,
                "received_at": result.received_at,
                "event": event_to_dict(normalized),
                "object_count": len(normalized.objects),
            },
        )

    def _post_heartbeat(self, start_response: Callable, environ: dict):
        payload = self._read_json_body(environ)
        heartbeat = parse_heartbeat_payload(payload)
        received_at = self.repository.record_heartbeat(heartbeat)
        LOGGER.info(
            "heartbeat received remote=%s device_id=%s status=%s received_at=%s",
            environ.get("REMOTE_ADDR", "-"),
            heartbeat.device_id,
            heartbeat.status,
            received_at,
        )
        return self._json(
            start_response,
            200,
            {
                "status": "ok",
                "result": "accepted",
                "received_at": received_at,
                "heartbeat": heartbeat_to_dict(heartbeat),
            },
        )

    def _html(self, start_response: Callable, body: str):
        payload = body.encode("utf-8")
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Content-Length", str(len(payload))),
            ],
        )
        return [payload]

    def _json(self, start_response: Callable, status_code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode("utf-8")
        start_response(
            f"{status_code} {self._status_text(status_code)}",
            [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    def _static_file(self, start_response: Callable, request_path: str):
        static_root = STATIC_DIR.resolve()
        relative_path = request_path.removeprefix("/static/").strip("/")
        target_path = (static_root / relative_path).resolve()
        if static_root not in target_path.parents and target_path != static_root:
            raise FileNotFoundError(request_path)
        if not target_path.is_file():
            raise FileNotFoundError(request_path)

        content_type, _ = mimetypes.guess_type(target_path.name)
        if content_type is None:
            content_type = "application/octet-stream"
        elif content_type.startswith("text/"):
            content_type = f"{content_type}; charset=utf-8"

        body = target_path.read_bytes()
        start_response(
            "200 OK",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    @staticmethod
    def _status_text(status_code: int) -> str:
        return {
            200: "OK",
            201: "Created",
            400: "Bad Request",
            409: "Conflict",
            404: "Not Found",
            500: "Internal Server Error",
        }.get(status_code, "OK")

    @staticmethod
    def _read_json_body(environ: dict) -> dict:
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            content_length = 0

        raw_body = environ["wsgi.input"].read(content_length)
        if not raw_body:
            raise SchemaValidationError("Request body is empty.")

        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError("Request body must contain valid JSON.") from exc


def _int_from_query(
    query: dict[str, list[str]],
    name: str,
    default: int,
    maximum: int | None = None,
) -> int:
    try:
        value = max(1, int(query.get(name, [default])[0]))
    except (TypeError, ValueError):
        return default
    if maximum is not None:
        return min(maximum, value)
    return value


def _format_timestamp_human(value: object) -> str:
    if not value:
        return "-"

    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text

    zone = parsed.tzname() or "UTC"
    return parsed.strftime("%d %b %Y, %H:%M:%S") + f" {zone}"


def _pretty_json(payload: object) -> str:
    return json.dumps(payload, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    repository = BrainRepository(DB_PATH)
    repository.initialize()
    if _seed_mock_enabled(os.getenv("BRAIN_SEED_MOCK")):
        LOGGER.info("Startup mock seeding is enabled.")
        seed_repository_if_empty(repository)
    else:
        LOGGER.info("Startup mock seeding is disabled.")
    application = BrainApplication(repository)

    host = os.getenv("BRAIN_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _port_from_env(os.getenv("BRAIN_PORT"))
    LOGGER.info("Central brain running on http://%s:%s", host, port)
    with make_server(host, port, application) as server:
        server.serve_forever()


def _port_from_env(raw_value: str | None) -> int:
    if raw_value in (None, ""):
        return 8000
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError("BRAIN_PORT must be an integer.") from exc
    if port < 1 or port > 65535:
        raise ValueError("BRAIN_PORT must be between 1 and 65535.")
    return port


def _seed_mock_enabled(raw_value: str | None) -> bool:
    if raw_value in (None, ""):
        return True
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("BRAIN_SEED_MOCK must be one of: 1, true, yes, on, 0, false, no, off.")


if __name__ == "__main__":
    main()
