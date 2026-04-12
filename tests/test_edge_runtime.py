from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib import error
from urllib.parse import urlparse
from unittest import mock
from wsgiref.util import setup_testing_defaults

import numpy as np

from brain.backend.app import BrainApplication
from brain.database.repository import BrainRepository
from brain.models.schema import parse_heartbeat_payload, parse_inference_payload
from edge.config import EdgeConfig, InspectionZoneConfig, ThresholdConfig
from edge.decision import DecisionEngine
from edge.filtering import DetectionFilter, is_bbox_center_in_zone
from edge.payloads import build_event_id, map_event_payload, map_heartbeat_payload
from edge.runtime import EdgeRuntime
from edge.stabilization import TrackStabilizer
from edge.tracking import TrackManager
from edge.transport import BrainTransport, TransportResult
from edge.types import BBox, ContaminationResult, DecisionResult, Detection, FinalizedInspection, FrameSample


class EdgeConfigAndPayloadTests(unittest.TestCase):
    def test_default_config_uses_canonical_endpoints(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = EdgeConfig.from_env()
        self.assertEqual(config.event_endpoint_url, "http://127.0.0.1:8000/api/inference")
        self.assertEqual(config.heartbeat_endpoint_url, "http://127.0.0.1:8000/api/heartbeat")
        self.assertEqual(config.allowed_classes, ("Metal",))
        self.assertEqual(config.thresholds.min_in_zone_frames_for_evaluation, 2)
        self.assertEqual(config.evaluation_zone, InspectionZoneConfig(x1=0.20, y1=0.55, x2=0.80, y2=0.95))
        self.assertFalse(config.debug.save_images)
        self.assertEqual(config.debug.output_dir.name, "edge_debug")

    def test_event_payload_matches_brain_v1_shape(self) -> None:
        inspection = FinalizedInspection(
            event_id=build_event_id(device_id="edge_demo_01", frame_index=123, track_number=7),
            device_id="edge_demo_01",
            source_type="edge_node",
            source_index=0,
            timestamp=datetime(2026, 3, 28, 11, 15, 30, tzinfo=UTC),
            frame_index=123,
            frame_width=1280,
            frame_height=720,
            object_id="track-0007",
            track_number=7,
            class_id=1,
            label="Metal",
            confidence=0.91,
            bbox=BBox(100, 120, 220, 260),
            decision=DecisionResult(
                decision="Accept",
                contamination_status="CLEAN",
                score=95,
                reason="dirty_probability<0.40",
            ),
            contamination=ContaminationResult(
                dirty_probability=0.12,
                clean_probability=0.88,
                applied=True,
            ),
            inspection_outcome={},
        )

        payload = map_event_payload(inspection)

        self.assertEqual(payload["schema_version"], "brain-v1")
        self.assertEqual(payload["event_type"], "inspection.finalized")
        self.assertEqual(payload["inspection_outcome"], {})
        self.assertEqual(payload["objects"][0]["bbox"], {"x1": 100, "y1": 120, "x2": 220, "y2": 260})
        self.assertEqual(payload["objects"][0]["decision"], "Accept")
        self.assertIn("refinement", payload["objects"][0])

        parsed = parse_inference_payload(payload)
        self.assertEqual(parsed.device_id, "edge_demo_01")
        self.assertEqual(parsed.objects[0].bbox, (100.0, 120.0, 220.0, 260.0))

    def test_heartbeat_payload_matches_brain_v1_shape(self) -> None:
        payload = map_heartbeat_payload(
            device_id="edge_demo_01",
            timestamp=datetime(2026, 3, 28, 11, 15, 45, tzinfo=UTC),
        )
        parsed = parse_heartbeat_payload(payload)
        self.assertEqual(parsed.device_id, "edge_demo_01")
        self.assertEqual(parsed.status, "online")


class FilteringAndDecisionTests(unittest.TestCase):
    def test_filter_enforces_confidence_class_and_size(self) -> None:
        frame = _make_frame(index=0, width=100, height=100)
        detections = [
            Detection(label="Metal", confidence=0.95, class_id=1, bbox=BBox(30, 30, 60, 60)),
            Detection(label="Metal", confidence=0.60, class_id=1, bbox=BBox(30, 30, 60, 60)),
            Detection(label="Glass", confidence=0.96, class_id=2, bbox=BBox(30, 30, 60, 60)),
            Detection(label="Metal", confidence=0.99, class_id=1, bbox=BBox(40, 40, 42, 42)),
        ]
        detection_filter = DetectionFilter(
            confidence_threshold=0.70,
            allowed_classes=("Metal",),
            min_size_ratio=0.01,
        )

        filtered = detection_filter.filter(frame, detections)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].label, "Metal")
        self.assertEqual(filtered[0].confidence, 0.95)

    def test_zone_helper_uses_bbox_center(self) -> None:
        zone = InspectionZoneConfig(x1=0.2, y1=0.55, x2=0.8, y2=0.95)
        self.assertTrue(
            is_bbox_center_in_zone(
                BBox(20, 50, 60, 90),
                frame_width=100,
                frame_height=100,
                zone=zone,
            )
        )
        self.assertFalse(
            is_bbox_center_in_zone(
                BBox(20, 20, 60, 60),
                frame_width=100,
                frame_height=100,
                zone=zone,
            )
        )

    def test_decision_engine_thresholds(self) -> None:
        engine = DecisionEngine(review_threshold=0.40, reject_threshold=0.70)

        self.assertEqual(
            engine.evaluate(ContaminationResult(dirty_probability=0.39, clean_probability=0.61, applied=True)).decision,
            "Accept",
        )
        self.assertEqual(
            engine.evaluate(ContaminationResult(dirty_probability=0.40, clean_probability=0.60, applied=True)).decision,
            "Review",
        )
        self.assertEqual(
            engine.evaluate(ContaminationResult(dirty_probability=0.70, clean_probability=0.30, applied=True)).decision,
            "Reject",
        )
        unavailable = engine.evaluate(ContaminationResult(applied=False, reason="missing_crop"))
        self.assertEqual(unavailable.decision, "Review")
        self.assertEqual(unavailable.contamination_status, "UNCERTAIN")


class TrackingLifecycleTests(unittest.TestCase):
    def test_stable_track_exits_after_missing_frames(self) -> None:
        manager = TrackManager(iou_threshold=0.30, max_missed_frames=2)
        stabilizer = TrackStabilizer(stable_after_frames=2, min_in_zone_frames_for_evaluation=2)
        detection = Detection(label="Metal", confidence=0.9, class_id=1, bbox=BBox(10, 10, 40, 40))

        active, finished = manager.update(_make_frame(index=0), [detection])
        self.assertFalse(finished)
        self.assertEqual(stabilizer.advance(active[0]), "tentative")

        active, finished = manager.update(_make_frame(index=1), [detection])
        self.assertFalse(finished)
        self.assertEqual(stabilizer.advance(active[0]), "stable")
        stabilizer.mark_evaluated(
            active[0],
            contamination=ContaminationResult(dirty_probability=0.2, clean_probability=0.8, applied=True),
            decision=DecisionResult("Accept", "CLEAN", 95, "dirty_probability<0.40"),
        )

        active, finished = manager.update(_make_frame(index=2), [])
        self.assertEqual(len(active), 1)
        self.assertFalse(finished)

        active, finished = manager.update(_make_frame(index=3), [])
        self.assertFalse(active)
        self.assertEqual(len(finished), 1)
        self.assertTrue(stabilizer.finish(finished[0]))
        self.assertEqual(finished[0].state, "exited")

    def test_tentative_track_expires_without_event(self) -> None:
        manager = TrackManager(iou_threshold=0.30, max_missed_frames=2)
        stabilizer = TrackStabilizer(stable_after_frames=2, min_in_zone_frames_for_evaluation=2)
        detection = Detection(label="Metal", confidence=0.9, class_id=1, bbox=BBox(10, 10, 40, 40))

        active, finished = manager.update(_make_frame(index=0), [detection])
        self.assertEqual(stabilizer.advance(active[0]), "tentative")

        manager.update(_make_frame(index=1), [])
        _, finished = manager.update(_make_frame(index=2), [])

        self.assertEqual(len(finished), 1)
        self.assertFalse(stabilizer.finish(finished[0]))
        self.assertEqual(finished[0].state, "expired")


class TransportTests(unittest.TestCase):
    def test_transport_accepts_duplicate_response(self) -> None:
        urlopen = ScriptedUrlopen(FakeResponse(200, {"result": "duplicate"}))
        transport = BrainTransport(
            event_endpoint_url="http://edge.local/api/inference",
            heartbeat_endpoint_url="http://edge.local/api/heartbeat",
            urlopen=urlopen,
            sleep_fn=lambda _: None,
        )

        result = transport.send_event({"hello": "world"})

        self.assertTrue(result.accepted)
        self.assertTrue(result.duplicate)
        self.assertEqual(urlopen.calls, 1)

    def test_transport_retries_transient_failure_then_accepts(self) -> None:
        urlopen = ScriptedUrlopen(
            error.URLError("brain unavailable"),
            FakeResponse(201, {"result": "accepted"}),
        )
        transport = BrainTransport(
            event_endpoint_url="http://edge.local/api/inference",
            heartbeat_endpoint_url="http://edge.local/api/heartbeat",
            urlopen=urlopen,
            sleep_fn=lambda _: None,
        )

        result = transport.send_event({"hello": "world"})

        self.assertTrue(result.accepted)
        self.assertEqual(urlopen.calls, 2)


class RuntimeTests(unittest.TestCase):
    def test_runtime_retries_same_event_id_until_accepted(self) -> None:
        frames = [_make_frame(index=index, width=100, height=100) for index in range(5)]
        detector = SequenceDetector(
            {
                0: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 20, 60, 60))],
                1: [Detection(label="Metal", confidence=0.92, class_id=1, bbox=BBox(20, 40, 60, 80))],
                2: [Detection(label="Metal", confidence=0.93, class_id=1, bbox=BBox(20, 45, 60, 85))],
                3: [Detection(label="Metal", confidence=0.91, class_id=1, bbox=BBox(20, 25, 60, 65))],
            }
        )
        transport = ScriptedTransport(
            [
                TransportResult(status_code=None, retryable=True, detail="temporary_failure"),
                TransportResult(status_code=201, accepted=True, payload={"result": "accepted"}),
            ]
        )
        config = _build_test_config()
        config.pending_retry_delay_seconds = 0.0
        runtime = EdgeRuntime(
            config,
            camera=FrameSequenceCamera(frames),
            detector=detector,
            contamination_evaluator=FixedContaminationEvaluator(),
            transport=transport,
        )

        runtime.run(max_frames=len(frames))

        self.assertEqual(len(transport.event_payloads), 2)
        self.assertEqual({payload["event_id"] for payload in transport.event_payloads}, {transport.event_payloads[0]["event_id"]})
        self.assertEqual(len(runtime.pending_events), 0)
        self.assertEqual(len(transport.heartbeat_payloads), 1)
        self.assertEqual(transport.event_payloads[0]["objects"][0]["decision"], "Accept")

    def test_runtime_posts_event_and_heartbeat_to_brain_server(self) -> None:
        frames = [_make_frame(index=index, width=100, height=100) for index in range(5)]
        detector = SequenceDetector(
            {
                0: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 20, 60, 60))],
                1: [Detection(label="Metal", confidence=0.92, class_id=1, bbox=BBox(20, 40, 60, 80))],
                2: [Detection(label="Metal", confidence=0.93, class_id=1, bbox=BBox(20, 45, 60, 85))],
                3: [Detection(label="Metal", confidence=0.91, class_id=1, bbox=BBox(20, 25, 60, 65))],
            }
        )

        with tempfile.TemporaryDirectory() as tempdir:
            repository = BrainRepository(Path(tempdir) / "brain.db")
            repository.initialize()
            application = BrainApplication(repository)

            base_url = "http://brain.local"
            transport = BrainTransport(
                event_endpoint_url=f"{base_url}/api/inference",
                heartbeat_endpoint_url=f"{base_url}/api/heartbeat",
                urlopen=WsgiUrlopen(application),
                sleep_fn=lambda _: None,
            )
            config = _build_test_config(base_url=base_url)
            runtime = EdgeRuntime(
                config,
                camera=FrameSequenceCamera(frames),
                detector=detector,
                contamination_evaluator=FixedContaminationEvaluator(),
                transport=transport,
            )
            runtime.run(max_frames=len(frames))

            self.assertEqual(repository.count_events(), 1)
            recent_event = repository.get_recent_events(limit=1)[0]
            overview = repository.get_overview()
            self.assertEqual(recent_event["device_id"], config.device_id)
            self.assertEqual(overview["active_devices"], 1)
            self.assertEqual(overview["devices"][0]["device_id"], config.device_id)

    def test_runtime_does_not_evaluate_before_zone_entry(self) -> None:
        frames = [_make_frame(index=index, width=100, height=100) for index in range(4)]
        detector = SequenceDetector(
            {
                0: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 10, 60, 50))],
                1: [Detection(label="Metal", confidence=0.91, class_id=1, bbox=BBox(20, 12, 60, 52))],
            }
        )
        evaluator = CountingContaminationEvaluator()
        transport = ScriptedTransport([])
        runtime = EdgeRuntime(
            _build_test_config(),
            camera=FrameSequenceCamera(frames),
            detector=detector,
            contamination_evaluator=evaluator,
            transport=transport,
        )

        runtime.run(max_frames=len(frames))

        self.assertEqual(evaluator.calls, 0)
        self.assertEqual(len(transport.event_payloads), 0)

    def test_runtime_emits_once_after_lingering_in_zone_then_exit(self) -> None:
        frames = [_make_frame(index=index, width=100, height=100) for index in range(7)]
        detector = SequenceDetector(
            {
                0: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 20, 60, 60))],
                1: [Detection(label="Metal", confidence=0.92, class_id=1, bbox=BBox(20, 40, 60, 80))],
                2: [Detection(label="Metal", confidence=0.93, class_id=1, bbox=BBox(20, 45, 60, 85))],
                3: [Detection(label="Metal", confidence=0.91, class_id=1, bbox=BBox(20, 46, 60, 86))],
                4: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 44, 60, 84))],
                5: [Detection(label="Metal", confidence=0.89, class_id=1, bbox=BBox(20, 24, 60, 64))],
            }
        )
        runtime = EdgeRuntime(
            _build_test_config(),
            camera=FrameSequenceCamera(frames),
            detector=detector,
            contamination_evaluator=FixedContaminationEvaluator(),
            transport=ScriptedTransport([]),
        )

        runtime.run(max_frames=len(frames))

        self.assertEqual(len(runtime.transport.event_payloads), 1)

    def test_runtime_emits_on_finish_if_evaluated_track_disappears_in_zone(self) -> None:
        frames = [_make_frame(index=index, width=100, height=100) for index in range(5)]
        detector = SequenceDetector(
            {
                0: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 20, 60, 60))],
                1: [Detection(label="Metal", confidence=0.92, class_id=1, bbox=BBox(20, 40, 60, 80))],
                2: [Detection(label="Metal", confidence=0.93, class_id=1, bbox=BBox(20, 45, 60, 85))],
            }
        )
        transport = ScriptedTransport([])
        runtime = EdgeRuntime(
            _build_test_config(),
            camera=FrameSequenceCamera(frames),
            detector=detector,
            contamination_evaluator=FixedContaminationEvaluator(),
            transport=transport,
        )

        runtime.run(max_frames=len(frames))

        self.assertEqual(len(transport.event_payloads), 1)

    def test_runtime_saves_debug_images_for_finalized_event(self) -> None:
        frames = [_make_frame(index=index, width=100, height=100) for index in range(7)]
        detector = SequenceDetector(
            {
                0: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 20, 60, 60))],
                1: [Detection(label="Metal", confidence=0.92, class_id=1, bbox=BBox(20, 40, 60, 80))],
                2: [Detection(label="Metal", confidence=0.93, class_id=1, bbox=BBox(20, 45, 60, 85))],
                3: [Detection(label="Metal", confidence=0.91, class_id=1, bbox=BBox(20, 46, 60, 86))],
                4: [Detection(label="Metal", confidence=0.90, class_id=1, bbox=BBox(20, 44, 60, 84))],
                5: [Detection(label="Metal", confidence=0.89, class_id=1, bbox=BBox(20, 24, 60, 64))],
            }
        )
        transport = ScriptedTransport([])
        with tempfile.TemporaryDirectory() as tempdir:
            config = _build_test_config()
            config.debug.save_images = True
            config.debug.output_dir = Path(tempdir)
            runtime = EdgeRuntime(
                config,
                camera=FrameSequenceCamera(frames),
                detector=detector,
                contamination_evaluator=FixedContaminationEvaluator(),
                transport=transport,
            )

            runtime.run(max_frames=len(frames))

            self.assertEqual(len(transport.event_payloads), 1)
            event_id = transport.event_payloads[0]["event_id"]
            saved_files = sorted(Path(tempdir).glob("*.png"))
            self.assertEqual(len(saved_files), 2)
            self.assertTrue(any(path.name.endswith("__crop.png") for path in saved_files))
            self.assertTrue(any(path.name.endswith("__frame.png") for path in saved_files))
            self.assertTrue(all(event_id in path.name for path in saved_files))
            self.assertTrue(all(path.stat().st_size > 0 for path in saved_files))


class FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class ScriptedUrlopen:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request, timeout: float = 0.0):
        del request, timeout
        self.calls += 1
        if not self.responses:
            raise AssertionError("Unexpected extra urlopen call.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FrameSequenceCamera:
    def __init__(self, frames: list[FrameSample]):
        self.frames = frames
        self.position = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def get_latest(self, *, after_index: int | None = None, timeout: float = 0.2) -> FrameSample | None:
        del after_index, timeout
        if self.position >= len(self.frames):
            return None
        frame = self.frames[self.position]
        self.position += 1
        return frame

    def stop(self) -> None:
        self.started = False


class SequenceDetector:
    def __init__(self, detections_by_frame: dict[int, list[Detection]]):
        self.detections_by_frame = detections_by_frame

    def detect(self, frame: FrameSample) -> list[Detection]:
        return list(self.detections_by_frame.get(frame.index, []))


class FixedContaminationEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, image, bbox: BBox) -> ContaminationResult:
        del image, bbox
        self.calls += 1
        return ContaminationResult(
            dirty_probability=0.12,
            clean_probability=0.88,
            applied=True,
        )


class CountingContaminationEvaluator(FixedContaminationEvaluator):
    pass


class ScriptedTransport:
    def __init__(self, event_results: list[TransportResult]):
        self.event_results = list(event_results)
        self.event_payloads: list[dict] = []
        self.heartbeat_payloads: list[dict] = []

    def send_event(self, payload: dict) -> TransportResult:
        self.event_payloads.append(payload)
        if self.event_results:
            return self.event_results.pop(0)
        return TransportResult(status_code=201, accepted=True, payload={"result": "accepted"})

    def send_heartbeat(self, payload: dict) -> TransportResult:
        self.heartbeat_payloads.append(payload)
        return TransportResult(status_code=200, accepted=True, payload={"result": "accepted"})


class WsgiUrlopen:
    def __init__(self, application) -> None:
        self.application = application

    def __call__(self, request, timeout: float = 0.0):
        del timeout
        parsed = urlparse(request.full_url)
        body = request.data or b""

        environ: dict[str, object] = {}
        setup_testing_defaults(environ)
        environ["REQUEST_METHOD"] = request.get_method()
        environ["PATH_INFO"] = parsed.path
        environ["QUERY_STRING"] = parsed.query
        environ["REMOTE_ADDR"] = "127.0.0.1"
        environ["CONTENT_LENGTH"] = str(len(body))
        environ["CONTENT_TYPE"] = request.headers.get("Content-Type", "application/json")
        environ["wsgi.input"] = io.BytesIO(body)

        state: dict[str, object] = {}

        def start_response(status: str, headers: list[tuple[str, str]]) -> None:
            state["status"] = status
            state["headers"] = headers

        response_body = b"".join(self.application(environ, start_response))
        status_text = str(state["status"])
        status_code = int(status_text.split(" ", 1)[0])

        if status_code >= 400:
            raise error.HTTPError(
                request.full_url,
                status_code,
                status_text,
                hdrs=None,
                fp=io.BytesIO(response_body),
            )

        return RawResponse(status=status_code, body=response_body)


class RawResponse:
    def __init__(self, *, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> RawResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _make_frame(index: int, *, width: int = 1280, height: int = 720) -> FrameSample:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    timestamp = datetime(2026, 3, 28, 10, 0, 0, tzinfo=UTC) + timedelta(seconds=index)
    return FrameSample.from_image(index=index, image=image, captured_at=timestamp)


def _build_test_config(*, base_url: str = "http://127.0.0.1:8000") -> EdgeConfig:
    return EdgeConfig(
        device_id="edge_demo_01",
        brain_base_url=base_url,
        event_endpoint_url=f"{base_url}/api/inference",
        heartbeat_endpoint_url=f"{base_url}/api/heartbeat",
        heartbeat_interval_seconds=15.0,
        pending_retry_delay_seconds=0.0,
        allowed_classes=("Metal",),
        evaluation_zone=InspectionZoneConfig(x1=0.2, y1=0.55, x2=0.8, y2=0.95),
        thresholds=ThresholdConfig(
            confidence=0.70,
            iou_match=0.30,
            stable_after_frames=2,
            max_missed_frames=2,
            min_in_zone_frames_for_evaluation=2,
            dirty_review_threshold=0.40,
            dirty_reject_threshold=0.70,
            min_size_ratio=None,
        ),
    )
