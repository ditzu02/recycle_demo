from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock
from wsgiref.util import setup_testing_defaults

from brain.backend.app import BrainApplication, _seed_mock_enabled
from brain.database.repository import BrainRepository
from brain.models.schema import (
    CANONICAL_EVENT_TYPE,
    CANONICAL_SCHEMA_VERSION,
    parse_heartbeat_payload,
    parse_inference_payload,
)


def build_event_payload(
    *,
    event_id: str = "evt-0001",
    device_id: str = "pi_01",
    timestamp: str = "2026-03-28T10:15:30Z",
) -> dict:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "event_type": CANONICAL_EVENT_TYPE,
        "event_id": event_id,
        "device_id": device_id,
        "timestamp": timestamp,
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
                "object_id": f"{event_id}-obj-01",
                "class_id": 2,
                "label": "Metal",
                "confidence": 0.91,
                "bbox": {
                    "x1": 100,
                    "y1": 120,
                    "x2": 220,
                    "y2": 260,
                },
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


def build_heartbeat_payload(
    *,
    device_id: str = "pi_01",
    timestamp: str = "2026-03-28T10:15:35Z",
    status: str = "online",
) -> dict:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "device_id": device_id,
        "timestamp": timestamp,
        "status": status,
    }


class BrainHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "brain.db"
        self.repository = BrainRepository(db_path)
        self.repository.initialize()
        self.application = BrainApplication(self.repository)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_parse_canonical_brain_v1_event(self) -> None:
        event = parse_inference_payload(build_event_payload())
        self.assertEqual(event.schema_version, CANONICAL_SCHEMA_VERSION)
        self.assertEqual(event.event_type, CANONICAL_EVENT_TYPE)
        self.assertEqual(event.event_uuid, "evt-0001")
        self.assertEqual(event.objects[0].object_id, "evt-0001-obj-01")
        self.assertEqual(event.objects[0].bbox, (100.0, 120.0, 220.0, 260.0))

    def test_parse_legacy_payload_still_supported(self) -> None:
        payload = {
            "device_id": "pi_legacy",
            "timestamp": "2026-03-28T10:15:30",
            "detections": [
                {
                    "label": "Glass",
                    "confidence": 0.73,
                    "bbox": [10, 20, 30, 40],
                }
            ],
        }
        event = parse_inference_payload(payload)
        self.assertEqual(event.schema_version, CANONICAL_SCHEMA_VERSION)
        self.assertEqual(event.event_type, CANONICAL_EVENT_TYPE)
        self.assertEqual(event.device_id, "pi_legacy")
        self.assertEqual(event.objects[0].bbox, (10.0, 20.0, 30.0, 40.0))

    def test_parse_heartbeat_payload(self) -> None:
        heartbeat = parse_heartbeat_payload(build_heartbeat_payload())
        self.assertEqual(heartbeat.device_id, "pi_01")
        self.assertEqual(heartbeat.status, "online")

    def test_repository_closes_connections_after_use(self) -> None:
        original_connect = sqlite3.connect
        opened_connections: list[sqlite3.Connection] = []

        class TrackingConnection(sqlite3.Connection):
            closed_count = 0

            def close(self) -> None:
                type(self).closed_count += 1
                super().close()

        def connect(*args, **kwargs):
            kwargs["factory"] = TrackingConnection
            connection = original_connect(*args, **kwargs)
            opened_connections.append(connection)
            return connection

        TrackingConnection.closed_count = 0
        with tempfile.TemporaryDirectory() as tempdir, mock.patch(
            "brain.database.repository.sqlite3.connect",
            side_effect=connect,
        ):
            repository = BrainRepository(Path(tempdir) / "brain.db")
            repository.initialize()
            repository.count_events()

        self.assertEqual(len(opened_connections), 3)
        self.assertEqual(TrackingConnection.closed_count, len(opened_connections))

    def test_seed_mock_env_values(self) -> None:
        self.assertTrue(_seed_mock_enabled(None))
        self.assertTrue(_seed_mock_enabled(""))
        self.assertTrue(_seed_mock_enabled("1"))
        self.assertTrue(_seed_mock_enabled("true"))
        self.assertFalse(_seed_mock_enabled("0"))
        self.assertFalse(_seed_mock_enabled("off"))

    def test_duplicate_event_returns_200_without_extra_rows(self) -> None:
        payload = build_event_payload()

        status_code, response = self._request_json("POST", "/api/inference", payload)
        self.assertEqual(status_code, 201)
        self.assertEqual(response["result"], "accepted")
        self.assertEqual(self.repository.count_events(), 1)

        status_code, response = self._request_json("POST", "/api/inference", payload)
        self.assertEqual(status_code, 200)
        self.assertEqual(response["result"], "duplicate")
        self.assertEqual(self.repository.count_events(), 1)

    def test_event_id_conflict_returns_409(self) -> None:
        payload = build_event_payload()
        self._request_json("POST", "/api/inference", payload)

        conflicting = build_event_payload(device_id="pi_02")
        status_code, response = self._request_json("POST", "/api/inference", conflicting)
        self.assertEqual(status_code, 409)
        self.assertIn("different device_id", response["detail"])

    def test_heartbeat_is_visible_before_first_event(self) -> None:
        status_code, response = self._request_json("POST", "/api/heartbeat", build_heartbeat_payload(device_id="pi_live"))
        self.assertEqual(status_code, 200)
        self.assertEqual(response["result"], "accepted")

        status_code, overview = self._request_json("GET", "/api/overview")
        self.assertEqual(status_code, 200)
        self.assertEqual(overview["active_devices"], 1)
        self.assertEqual(overview["devices"][0]["device_id"], "pi_live")
        self.assertEqual(overview["devices"][0]["last_contact_kind"], "heartbeat")
        self.assertEqual(overview["devices"][0]["heartbeat_freshness"], "fresh")
        self.assertEqual(overview["devices"][0]["device_state"], "online")
        self.assertEqual(overview["devices"][0]["event_count"], 0)

    def test_device_state_is_derived_from_heartbeat_age_and_status(self) -> None:
        self._request_json("POST", "/api/inference", build_event_payload(device_id="pi_state"))

        status_code, overview = self._request_json("GET", "/api/overview")
        self.assertEqual(status_code, 200)
        self.assertEqual(overview["heartbeat_thresholds"]["fresh_seconds"], 30)
        self.assertEqual(overview["heartbeat_thresholds"]["offline_seconds"], 90)
        self.assertEqual(overview["devices"][0]["heartbeat_freshness"], "never")
        self.assertEqual(overview["devices"][0]["device_state"], "unknown")

        self._request_json("POST", "/api/heartbeat", build_heartbeat_payload(device_id="pi_state", status="online"))
        status_code, overview = self._request_json("GET", "/api/overview")
        self.assertEqual(status_code, 200)
        self.assertEqual(overview["devices"][0]["heartbeat_freshness"], "fresh")
        self.assertEqual(overview["devices"][0]["device_state"], "online")

        self._set_heartbeat_state(
            device_id="pi_state",
            received_at=(datetime.now(UTC) - timedelta(seconds=45)).isoformat(),
            status="online",
        )
        status_code, overview = self._request_json("GET", "/api/overview")
        self.assertEqual(status_code, 200)
        self.assertEqual(overview["devices"][0]["heartbeat_freshness"], "stale")
        self.assertEqual(overview["devices"][0]["device_state"], "stale")

        self._set_heartbeat_state(
            device_id="pi_state",
            received_at=(datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
            status="online",
        )
        status_code, overview = self._request_json("GET", "/api/overview")
        self.assertEqual(status_code, 200)
        self.assertEqual(overview["devices"][0]["device_state"], "offline")

        self._set_heartbeat_state(
            device_id="pi_state",
            received_at=datetime.now(UTC).isoformat(),
            status="offline",
        )
        status_code, overview = self._request_json("GET", "/api/overview")
        self.assertEqual(status_code, 200)
        self.assertEqual(overview["devices"][0]["heartbeat_freshness"], "fresh")
        self.assertEqual(overview["devices"][0]["device_state"], "offline")

    def test_events_are_ordered_by_receive_time(self) -> None:
        first = build_event_payload(event_id="evt-recent-ts", timestamp="2026-03-28T10:30:00Z")
        second = build_event_payload(event_id="evt-old-ts", timestamp="2026-03-28T09:00:00Z")

        self._request_json("POST", "/api/inference", first)
        self._request_json("POST", "/api/inference", second)

        status_code, payload = self._request_json("GET", "/api/events?limit=10")
        self.assertEqual(status_code, 200)
        self.assertEqual(payload["events"][0]["event_uuid"], "evt-old-ts")
        self.assertEqual(payload["events"][1]["event_uuid"], "evt-recent-ts")

    def test_live_overview_endpoint_returns_compact_recent_stream(self) -> None:
        first = build_event_payload(event_id="evt-live-001", device_id="pi_01")
        second = build_event_payload(event_id="evt-live-002", device_id="pi_02")

        self._request_json("POST", "/api/inference", first)
        self._request_json("POST", "/api/inference", second)

        status_code, payload = self._request_json("GET", "/api/overview/live?limit=1")
        self.assertEqual(status_code, 200)
        self.assertEqual(payload["limit"], 1)
        self.assertEqual(payload["summary"]["total_events"], 2)
        self.assertEqual(payload["summary"]["total_objects"], 2)
        self.assertEqual(payload["summary"]["active_devices"], 2)
        self.assertEqual(payload["summary"]["accept_count"], 2)
        self.assertIn("devices_html", payload)
        self.assertIn("recent_devices_html", payload)
        self.assertIn("data-device-id=\"pi_02\"", payload["devices_html"])
        self.assertIn("UNKNOWN", payload["devices_html"])
        self.assertEqual(len(payload["items"]), 1)

        item = payload["items"][0]
        self.assertEqual(item["event_uuid"], "evt-live-002")
        self.assertEqual(item["device_id"], "pi_02")
        self.assertEqual(item["label"], "Metal")
        self.assertEqual(item["decision"], "Accept")
        self.assertEqual(item["confidence"], 0.91)
        self.assertEqual(
            set(item),
            {
                "id",
                "event_uuid",
                "object_id",
                "device_id",
                "label",
                "decision",
                "confidence",
                "timestamp",
                "received_at",
            },
        )

    def test_validation_failure_returns_400(self) -> None:
        invalid_payload = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "event_type": CANONICAL_EVENT_TYPE,
            "event_id": "evt-invalid",
            "device_id": "pi_01",
            "timestamp": "not-a-timestamp",
            "source": {},
            "frame": {},
            "inspection_outcome": {},
            "objects": [],
        }
        status_code, response = self._request_json("POST", "/api/inference", invalid_payload)
        self.assertEqual(status_code, 400)
        self.assertIn("timestamp", response["detail"])

    def test_dashboard_pages_render(self) -> None:
        self._request_json("POST", "/api/inference", build_event_payload())
        self._request_json("POST", "/api/heartbeat", build_heartbeat_payload())

        status_code, overview_html = self._request_html("GET", "/")
        self.assertEqual(status_code, 200)
        self.assertIn("DETECTED OBJECTS", overview_html)
        self.assertIn("DEVICES", overview_html)
        self.assertIn("LIVE STREAM", overview_html)
        self.assertNotIn("Total events", overview_html)
        self.assertNotIn("Recent device activity", overview_html)
        self.assertIn("LAST CONTACT", overview_html)
        self.assertIn("heartbeat", overview_html.lower())

        status_code, objects_html = self._request_html("GET", "/events")
        self.assertEqual(status_code, 200)
        self.assertIn("RESULTS", objects_html)
        self.assertIn("RECEIVED", objects_html)
        self.assertIn("DEVICE TIME", objects_html)
        self.assertNotIn("Latest inference events", objects_html)
        self.assertNotIn("Latest detected objects", objects_html)

        status_code, system_html = self._request_html("GET", "/api")
        self.assertEqual(status_code, 200)
        self.assertIn("INTERFACE", system_html)
        self.assertIn("PAYLOAD", system_html)
        self.assertIn("FLOW", system_html)
        self.assertIn("/api/inference", system_html)
        self.assertNotIn("Brain-v1 API explorer", system_html)
        self.assertNotIn("Distributed edge-AI coordination", system_html)

    def _set_heartbeat_state(self, device_id: str, received_at: str, status: str) -> None:
        connection = sqlite3.connect(self.repository.db_path)
        try:
            connection.execute(
                """
                UPDATE device_status
                SET
                    last_heartbeat_received_at = ?,
                    last_heartbeat_status = ?
                WHERE device_id = ?
                """,
                (received_at, status, device_id),
            )
            connection.commit()
        finally:
            connection.close()

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        environ: dict[str, object] = {}
        setup_testing_defaults(environ)
        environ["REQUEST_METHOD"] = method
        if "?" in path:
            request_path, query_string = path.split("?", 1)
        else:
            request_path, query_string = path, ""
        environ["PATH_INFO"] = request_path
        environ["QUERY_STRING"] = query_string
        environ["REMOTE_ADDR"] = "127.0.0.1"

        body = b""
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            environ["CONTENT_TYPE"] = "application/json"
        environ["CONTENT_LENGTH"] = str(len(body))
        environ["wsgi.input"] = io.BytesIO(body)

        state: dict[str, object] = {}

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            state["status"] = status
            state["headers"] = headers

        response_body = b"".join(self.application(environ, start_response))
        status_code = int(str(state["status"]).split(" ", 1)[0])
        return status_code, json.loads(response_body.decode("utf-8"))

    def _request_html(self, method: str, path: str) -> tuple[int, str]:
        environ: dict[str, object] = {}
        setup_testing_defaults(environ)
        environ["REQUEST_METHOD"] = method
        environ["PATH_INFO"] = path
        environ["QUERY_STRING"] = ""
        environ["REMOTE_ADDR"] = "127.0.0.1"
        environ["CONTENT_LENGTH"] = "0"
        environ["wsgi.input"] = io.BytesIO(b"")

        state: dict[str, object] = {}

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            state["status"] = status
            state["headers"] = headers

        response_body = b"".join(self.application(environ, start_response))
        status_code = int(str(state["status"]).split(" ", 1)[0])
        return status_code, response_body.decode("utf-8")


if __name__ == "__main__":
    unittest.main()
