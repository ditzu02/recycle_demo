from __future__ import annotations

import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from jinja2 import Environment, FileSystemLoader, select_autoescape

from brain.database.repository import BrainRepository
from brain.mock.seed import seed_repository_if_empty
from brain.models.schema import SchemaValidationError, event_to_dict, parse_inference_payload


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "brain.db"


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
            if method == "GET" and path.startswith("/static/"):
                return self._static_file(start_response, path)
        except SchemaValidationError as exc:
            return self._json(start_response, 400, {"status": "error", "detail": str(exc)})
        except FileNotFoundError:
            return self._json(start_response, 404, {"status": "error", "detail": "Not found"})
        except Exception as exc:  # pragma: no cover - demo-level fallback
            return self._json(start_response, 500, {"status": "error", "detail": str(exc)})

        return self._json(start_response, 404, {"status": "error", "detail": "Not found"})

    def _overview_page(self, start_response: Callable):
        overview = self.repository.get_overview(device_limit=8, recent_device_limit=8)
        template = self.templates.get_template("overview.html")
        body = template.render(
            title="Recycle Brain | Overview",
            page_name="Overview",
            page_description="Central monitoring dashboard for device activity, object decisions, and recent edge-node output.",
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
            page_description="Recent inference traffic and object-level decisions reported by the distributed edge-AI nodes.",
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
            "device_id": "pi_01",
            "timestamp": "2026-01-01T12:00:00",
            "objects": [
                {
                    "label": "Metal",
                    "confidence": 0.91,
                    "score": 87,
                    "decision": "Accept",
                    "bbox": [100, 120, 220, 260],
                    "dirty_probability": 0.12,
                }
            ],
        }
        template = self.templates.get_template("api.html")
        body = template.render(
            title="Recycle Brain | API",
            page_name="API",
            page_description="System endpoints, payload examples, and live response samples for the local central brain service.",
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
                    "description": "Aggregated dashboard totals, class counts, and per-device summaries.",
                    "notes": "Used by the Overview dashboard page.",
                    "sample_label": "Live response sample",
                    "sample_body": _pretty_json(overview),
                },
                {
                    "method": "GET",
                    "path": "/api/events?limit=20",
                    "description": "Recent event rows and object-level results for monitoring tables.",
                    "notes": f"Query parameter `limit` controls row count. Sample below uses {events_sample_limit}.",
                    "sample_label": "Live response sample",
                    "sample_body": _pretty_json(events_payload),
                },
                {
                    "method": "POST",
                    "path": "/api/inference",
                    "description": "Receives edge-node inference payloads and stores normalized event/object records.",
                    "notes": "Accepts the simple mock payload today and the richer future edge-node shape already supported by the schema layer.",
                    "sample_label": "Sample request body",
                    "sample_body": _pretty_json(sample_inference_payload),
                },
            ],
        )
        return self._html(start_response, body)

    def _post_inference(self, start_response: Callable, environ: dict):
        try:
            content_length = int(environ.get("CONTENT_LENGTH") or "0")
        except ValueError:
            content_length = 0

        raw_body = environ["wsgi.input"].read(content_length)
        if not raw_body:
            raise SchemaValidationError("Request body is empty.")

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError("Request body must contain valid JSON.") from exc
        normalized = parse_inference_payload(payload)
        event_id = self.repository.insert_event(normalized)

        # TODO: add device authentication and per-device registration once real Raspberry Pi nodes are online.
        return self._json(
            start_response,
            201,
            {
                "status": "ok",
                "stored_event_id": event_id,
                "event": event_to_dict(normalized),
                "object_count": len(normalized.objects),
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
            404: "Not Found",
            500: "Internal Server Error",
        }.get(status_code, "OK")


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
    repository = BrainRepository(DB_PATH)
    repository.initialize()
    seed_repository_if_empty(repository)
    application = BrainApplication(repository)

    host = "127.0.0.1"
    port = 8000
    print(f"Central brain running on http://{host}:{port}")
    with make_server(host, port, application) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
