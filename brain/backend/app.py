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

from brain.database.repository import (
    HEARTBEAT_FRESH_SECONDS,
    HEARTBEAT_OFFLINE_SECONDS,
    BrainRepository,
    EventConflictError,
)
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
LIVE_STREAM_REFRESH_SECONDS = 3
LIVE_STREAM_LIMIT = 10
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
            if method == "GET" and path == "/api/overview/live":
                limit = _int_from_query(query, "limit", default=LIVE_STREAM_LIMIT, maximum=25)
                overview = self.repository.get_overview(device_limit=8, recent_device_limit=8)
                return self._json(
                    start_response,
                    200,
                    {
                        "summary": _overview_summary(overview),
                        "items": self.repository.get_live_inference_stream(limit=limit),
                        "limit": limit,
                        "devices_html": self._render_overview_device_accordion(overview["devices"]),
                        "recent_devices_html": self._render_overview_recent_devices(overview["recent_devices"]),
                    },
                )
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
            title="Brain | Overview",
            page_name="OVERVIEW",
            page_description="",
            active_page="overview",
            overview=overview,
            live_stream_limit=LIVE_STREAM_LIMIT,
            recent_objects=self.repository.get_live_inference_stream(limit=LIVE_STREAM_LIMIT),
        )
        return self._html(start_response, body)

    def _render_overview_device_accordion(self, devices: object) -> str:
        template = self.templates.get_template("_overview_device_accordion.html")
        return template.render(devices=devices)

    def _render_overview_recent_devices(self, recent_devices: object) -> str:
        template = self.templates.get_template("_overview_recent_devices.html")
        return template.render(recent_devices=recent_devices)

    def _events_page(self, start_response: Callable, query: dict[str, list[str]]):
        page = _int_from_query(query, "page", default=1)
        limit = _int_from_query(query, "limit", default=25, maximum=100)
        offset = (page - 1) * limit
        objects = self.repository.get_recent_object_results(limit=limit + 1, offset=offset)
        has_next = len(objects) > limit
        objects = objects[:limit]
        template = self.templates.get_template("events.html")
        body = template.render(
            title="Brain | Events",
            page_name="EVENTS",
            page_description="",
            active_page="events",
            objects=objects,
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
                "frame_index": 42,
            },
            "objects": [
                {
                    "object_id": "obj-0001",
                    "label": "Metal",
                    "confidence": 0.91,
                    "decision": "Accept",
                    "contamination_status": "CLEAN",
                    "dirty_probability": 0.12,
                }
            ],
        }
        template = self.templates.get_template("api.html")
        body = template.render(
            title="Brain | System",
            page_name="SYSTEM",
            page_description="",
            active_page="api",
            summary_cards=[
                {
                    "label": "MODE",
                    "value": "LOCAL",
                    "detail": "WSGI / SQLITE",
                },
                {
                    "label": "DEVICES",
                    "value": overview["active_devices"],
                    "detail": "SEEN",
                },
                {
                    "label": "OBJECTS",
                    "value": overview["total_objects"],
                    "detail": "STORED",
                },
                {
                    "label": "ACCEPT",
                    "value": overview["accept_count"],
                    "detail": "CURRENT",
                },
            ],
            system_flow=[
                {
                    "title": "EDGE",
                    "description": "LOCAL INFERENCE",
                },
                {
                    "title": "INGEST",
                    "description": "JSON VALIDATION",
                },
                {
                    "title": "STORE",
                    "description": "SQLITE",
                },
                {
                    "title": "VIEW",
                    "description": "POLLING UI",
                },
            ],
            payload_example=_pretty_json(sample_inference_payload),
            payload_points=[
                "ONE OBJECT / REQUEST",
                "EDGE INFERENCE STAYS LOCAL",
                "EDGE TIME + BRAIN TIME",
            ],
            system_notes=[
                "MULTI-DEVICE COORDINATION",
                "IDEMPOTENT RESULT IDS",
                "LOCAL ONLY",
            ],
            endpoints=[
                {
                    "method": "GET",
                    "path": "/health",
                    "role": "CHECK",
                    "description": "PROCESS",
                    "sample_label": "RESP",
                    "sample_body": _pretty_json({"status": "ok"}),
                },
                {
                    "method": "POST",
                    "path": "/api/inference",
                    "role": "WRITE",
                    "description": "RESULT",
                    "sample_label": "RESP",
                    "sample_body": _pretty_json(
                        {
                            "status": "ok",
                            "result": "accepted",
                            "event_id": "pi_01-final-0001",
                            "object_count": 1,
                        }
                    ),
                },
                {
                    "method": "POST",
                    "path": "/api/heartbeat",
                    "role": "LIVENESS",
                    "description": "DEVICE STATE",
                    "sample_label": "RESP",
                    "sample_body": _pretty_json(
                        {
                            "status": "ok",
                            "result": "accepted",
                            "device_id": "pi_01",
                        }
                    ),
                },
                {
                    "method": "GET",
                    "path": "/api/overview",
                    "role": "READ",
                    "description": "SUMMARY",
                    "sample_label": "SHAPE",
                    "sample_body": _pretty_json(
                        {
                            "total_objects": overview["total_objects"],
                            "active_devices": overview["active_devices"],
                            "devices": ["..."],
                        }
                    ),
                },
                {
                    "method": "GET",
                    "path": "/api/overview/live?limit=10",
                    "role": "LIVE",
                    "description": "STREAM",
                    "sample_label": "SHAPE",
                    "sample_body": _pretty_json(
                        {
                            "summary": {
                                "total_objects": overview["total_objects"],
                            },
                            "items": [
                                {
                                    "device_id": "pi_01",
                                    "label": "Metal",
                                    "decision": "Accept",
                                    "confidence": 0.91,
                                }
                            ],
                        }
                    ),
                },
                {
                    "method": "GET",
                    "path": "/api/events?limit=20",
                    "role": "READ",
                    "description": "RESULTS",
                    "sample_label": "SHAPE",
                    "sample_body": _pretty_json(
                        {
                            "objects": [{"label": "...", "decision": "..."}],
                        }
                    ),
                },
            ],
            dashboard_refresh_seconds=DASHBOARD_REFRESH_SECONDS,
            live_stream_refresh_seconds=LIVE_STREAM_REFRESH_SECONDS,
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


def _overview_summary(overview: dict[str, object]) -> dict[str, object]:
    return {
        "total_events": overview["total_events"],
        "total_objects": overview["total_objects"],
        "active_devices": overview["active_devices"],
        "accept_count": overview["accept_count"],
        "review_count": overview["review_count"],
        "reject_count": overview["reject_count"],
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    heartbeat_fresh_seconds = _positive_int_from_env(
        os.getenv("BRAIN_HEARTBEAT_FRESH_SECONDS"),
        default=HEARTBEAT_FRESH_SECONDS,
        name="BRAIN_HEARTBEAT_FRESH_SECONDS",
    )
    heartbeat_offline_seconds = _positive_int_from_env(
        os.getenv("BRAIN_HEARTBEAT_OFFLINE_SECONDS"),
        default=HEARTBEAT_OFFLINE_SECONDS,
        name="BRAIN_HEARTBEAT_OFFLINE_SECONDS",
    )
    repository = BrainRepository(
        DB_PATH,
        heartbeat_fresh_seconds=heartbeat_fresh_seconds,
        heartbeat_offline_seconds=heartbeat_offline_seconds,
    )
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


def _positive_int_from_env(raw_value: str | None, default: int, name: str) -> int:
    if raw_value in (None, ""):
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


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
