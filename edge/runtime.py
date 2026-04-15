from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from edge.config import EdgeConfig
from edge.decision import DecisionEngine
from edge.filtering import DetectionFilter, is_bbox_center_in_zone, zone_bounds_to_pixels
from edge.payloads import build_event_id, map_event_payload, map_heartbeat_payload
from edge.stabilization import TrackStabilizer
from edge.tracking import TrackManager
from edge.transport import BrainTransport
from edge.types import BBox, ContaminationResult, FinalizedInspection, TrackState


LOGGER = logging.getLogger(__name__)


@dataclass
class PendingEvent:
    inspection: FinalizedInspection
    track: TrackState
    next_attempt_at: float = 0.0


class EdgeRuntime:
    def __init__(
        self,
        config: EdgeConfig,
        *,
        camera=None,
        detector=None,
        contamination_evaluator=None,
        transport: BrainTransport | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.config.validate()
        self.camera = camera
        self.detector = detector
        self.contamination_evaluator = contamination_evaluator
        self.transport = transport or BrainTransport(
            event_endpoint_url=config.event_endpoint_url,
            heartbeat_endpoint_url=config.heartbeat_endpoint_url,
            timeout_seconds=config.request_timeout_seconds,
            event_retry_attempts=config.event_retry_attempts,
            event_retry_backoff_seconds=config.event_retry_backoff_seconds,
        )
        self.monotonic = monotonic or time.monotonic
        self.filter = DetectionFilter(
            confidence_threshold=config.thresholds.confidence,
            allowed_classes=config.allowed_classes,
            min_size_ratio=config.thresholds.min_size_ratio,
        )
        self.tracks = TrackManager(
            iou_threshold=config.thresholds.iou_match,
            max_missed_frames=config.thresholds.max_missed_frames,
        )
        self.stabilizer = TrackStabilizer(
            stable_after_frames=config.thresholds.stable_after_frames,
            min_in_zone_frames_for_evaluation=config.thresholds.min_in_zone_frames_for_evaluation,
        )
        self.decision_engine = DecisionEngine(
            review_threshold=config.thresholds.dirty_review_threshold,
            reject_threshold=config.thresholds.dirty_reject_threshold,
            label_accept_confidence=config.thresholds.label_accept_confidence,
        )
        self.session_identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.pending_events: dict[str, PendingEvent] = {}
        self._last_heartbeat_at = 0.0

    def run(self, *, max_frames: int | None = None) -> None:
        self._ensure_components()
        processed_frames = 0
        last_frame_index = -1
        self.camera.start()
        self._send_heartbeat(force=True)

        try:
            while True:
                self._flush_pending_events()
                self._send_heartbeat()

                frame = self.camera.get_latest(after_index=last_frame_index, timeout=0.2)
                if frame is None:
                    if max_frames is not None and processed_frames >= max_frames:
                        break
                    continue

                last_frame_index = frame.index
                processed_frames += 1
                self._process_frame(frame)

                if self.config.show_preview and self._render_preview(frame):
                    break
                if max_frames is not None and processed_frames >= max_frames:
                    break

            self._flush_pending_events(force=True)
        finally:
            self.camera.stop()
            if self.config.show_preview:
                import cv2

                cv2.destroyAllWindows()

    def _ensure_components(self) -> None:
        if self.camera is None:
            from edge.camera import CameraCapture

            self.camera = CameraCapture(self.config.camera)
        if self.detector is None:
            from edge.detection import YOLODetector

            self.detector = YOLODetector(
                model_path=str(self.config.models.yolo_model_path),
                image_size=self.config.models.yolo_image_size,
                confidence_floor=self.config.models.yolo_confidence_floor,
                class_names=self.config.models.yolo_class_names,
            )
        if self.contamination_evaluator is None:
            if self._supports_metal_refinement():
                from edge.contamination import MetalContaminationEvaluator

                self.contamination_evaluator = MetalContaminationEvaluator(
                    weights_path=str(self.config.models.contamination_model_path),
                )

    def _process_frame(self, frame) -> None:
        detections = self.detector.detect(frame)
        filtered = self.filter.filter(frame, detections)
        active_tracks, finished_tracks = self.tracks.update(frame, filtered)

        for track in active_tracks:
            just_left_zone = False
            if track.was_observed_in_frame(frame.index):
                just_left_zone = self._update_track_zone(track, frame)
            self.stabilizer.advance(track)
            if self.stabilizer.should_evaluate(track):
                self._evaluate_track(track)
            if just_left_zone and self._has_pending_contamination_samples(track):
                self._complete_metal_evaluation(track)
            if self.stabilizer.should_emit_on_zone_exit(track, just_left_zone=just_left_zone):
                self._queue_finalized_event(track)

        for track in finished_tracks:
            if self._has_pending_contamination_samples(track):
                self._complete_metal_evaluation(track)
            should_finalize = self.stabilizer.finish(track)
            if not should_finalize:
                continue
            self._queue_finalized_event(track)

    def _evaluate_track(self, track: TrackState) -> None:
        if self._is_metal_label(track.label):
            sample_possible = self._sample_metal_contamination(track)
            if not sample_possible and not track.contamination_samples:
                self._complete_metal_evaluation(track)
                return
            if len(track.contamination_samples) < self.config.thresholds.contamination_sample_count:
                return
            self._complete_metal_evaluation(track)
            return

        snapshot = track.best_in_zone_snapshot or track.best_snapshot or track.latest_snapshot
        label_confidence = track.confidence if snapshot is None else snapshot.confidence
        contamination: ContaminationResult | None = None
        contamination = self.decision_engine.canonicalize_contamination(label=track.label, contamination=contamination)
        decision = self.decision_engine.evaluate(
            label=track.label,
            confidence=label_confidence,
            contamination=contamination,
        )
        self.stabilizer.mark_evaluated(track, contamination=contamination, decision=decision)

    def _sample_metal_contamination(self, track: TrackState) -> bool:
        snapshot = track.latest_in_zone_snapshot or track.best_in_zone_snapshot
        if (
            snapshot is None
            or snapshot.image is None
            or self.contamination_evaluator is None
        ):
            return False

        if snapshot.frame_index in track.contamination_sample_frame_indexes:
            return True

        sample = self._evaluate_contamination_crop(snapshot.image)
        track.contamination_samples.append(sample)
        track.contamination_sample_frame_indexes.add(snapshot.frame_index)
        return True

    def _complete_metal_evaluation(self, track: TrackState) -> None:
        snapshot = track.best_in_zone_snapshot or track.best_snapshot or track.latest_snapshot
        label_confidence = track.confidence if snapshot is None else snapshot.confidence
        contamination = self._average_contamination_samples(track)
        contamination = self.decision_engine.canonicalize_contamination(label=track.label, contamination=contamination)
        decision = self.decision_engine.evaluate(
            label=track.label,
            confidence=label_confidence,
            contamination=contamination,
        )
        self.stabilizer.mark_evaluated(track, contamination=contamination, decision=decision)

    def _average_contamination_samples(self, track: TrackState) -> ContaminationResult:
        available_samples = [
            sample
            for sample in track.contamination_samples
            if sample.applied and sample.available
        ]
        if not available_samples:
            return ContaminationResult(applied=False, reason="contamination_samples_unavailable")

        sample_count = len(available_samples)
        dirty_probability = sum(sample.dirty_probability or 0.0 for sample in available_samples) / sample_count
        clean_probability = sum(sample.clean_probability or 0.0 for sample in available_samples) / sample_count
        return ContaminationResult(
            dirty_probability=dirty_probability,
            clean_probability=clean_probability,
            applied=True,
            reason=f"averaged_{sample_count}_contamination_samples",
        )

    def _has_pending_contamination_samples(self, track: TrackState) -> bool:
        return (
            track.state == "stable"
            and track.decision is None
            and self._is_metal_label(track.label)
            and bool(track.contamination_samples)
        )

    def _finalize_track(self, track: TrackState) -> FinalizedInspection:
        snapshot = track.best_in_zone_snapshot or track.best_snapshot or track.latest_snapshot
        if snapshot is None or track.decision is None:
            raise RuntimeError("Cannot finalize a track without a snapshot and decision.")

        if track.event_id is None:
            # The event_id is generated once from the finalized snapshot and track number
            # so retries never create a second ingest key for the same object lifecycle.
            track.event_id = build_event_id(
                device_id=self.config.device_id,
                frame_index=snapshot.frame_index,
                track_number=track.track_number,
                session_identifier=self.session_identifier,
            )

        return FinalizedInspection(
            event_id=track.event_id,
            device_id=self.config.device_id,
            source_type=self.config.source_type,
            source_index=self.config.source_index,
            timestamp=snapshot.captured_at,
            frame_index=snapshot.frame_index,
            frame_width=snapshot.frame_width,
            frame_height=snapshot.frame_height,
            object_id=track.object_id,
            track_number=track.track_number,
            class_id=snapshot.class_id,
            label=snapshot.label,
            confidence=snapshot.confidence,
            bbox=snapshot.bbox,
            decision=track.decision,
            contamination=track.contamination,
            inspection_outcome={},
        )

    def _queue_finalized_event(self, track: TrackState) -> None:
        inspection = self._finalize_track(track)
        if inspection.event_id in self.pending_events:
            return
        self._save_debug_images(inspection, track)
        self.stabilizer.mark_event_queued(track)
        self.pending_events[inspection.event_id] = PendingEvent(inspection=inspection, track=track)
        self._flush_pending_events(force=True)

    def _update_track_zone(self, track: TrackState, frame) -> bool:
        snapshot = track.latest_snapshot
        if snapshot is None or snapshot.frame_index != frame.index:
            return False

        in_zone = is_bbox_center_in_zone(
            snapshot.bbox,
            frame_width=snapshot.frame_width,
            frame_height=snapshot.frame_height,
            zone=self.config.evaluation_zone,
        )
        return track.update_evaluation_zone(
            frame=frame,
            in_zone=in_zone,
            store_frame_image=self.config.debug.save_images and self.config.debug.save_annotated_frame,
        )

    def _flush_pending_events(self, *, force: bool = False) -> None:
        now = self.monotonic()
        for event_id in list(self.pending_events.keys()):
            pending = self.pending_events[event_id]
            if not force and now < pending.next_attempt_at:
                continue

            result = self.transport.send_event(map_event_payload(pending.inspection))
            if result.accepted:
                self.stabilizer.mark_emitted(pending.track)
                self.pending_events.pop(event_id, None)
                LOGGER.info("event sent device_id=%s event_id=%s duplicate=%s", self.config.device_id, event_id, result.duplicate)
                continue

            if result.retryable:
                pending.next_attempt_at = now + self.config.pending_retry_delay_seconds
                LOGGER.warning("event send retry scheduled event_id=%s detail=%s", event_id, result.detail)
                continue

            LOGGER.error("dropping non-retryable event event_id=%s status=%s detail=%s", event_id, result.status_code, result.detail)
            self.pending_events.pop(event_id, None)

    def _send_heartbeat(self, *, force: bool = False) -> None:
        now = self.monotonic()
        if not force and now - self._last_heartbeat_at < self.config.heartbeat_interval_seconds:
            return
        payload = map_heartbeat_payload(
            device_id=self.config.device_id,
            timestamp=datetime.now(UTC),
            status="online",
        )
        result = self.transport.send_heartbeat(payload)
        self._last_heartbeat_at = now
        if not result.accepted:
            LOGGER.warning("heartbeat send failed device_id=%s detail=%s", self.config.device_id, result.detail)

    def _evaluate_contamination_crop(self, crop) -> ContaminationResult:
        evaluate_crop = getattr(self.contamination_evaluator, "evaluate_crop", None)
        if evaluate_crop is not None:
            return evaluate_crop(crop)

        height, width = crop.shape[:2]
        return self.contamination_evaluator.evaluate(crop, BBox(0, 0, width, height))

    def _save_debug_images(self, inspection: FinalizedInspection, track: TrackState) -> None:
        if not self.config.debug.save_images:
            return

        snapshot = track.best_in_zone_snapshot or track.best_snapshot or track.latest_snapshot
        if snapshot is None or snapshot.image is None:
            LOGGER.warning("debug image save skipped event_id=%s detail=missing_snapshot_image", inspection.event_id)
            return

        output_dir = self.config.debug.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = self._build_debug_file_stem(inspection)

        try:
            self._write_debug_image(output_dir / f"{file_stem}__crop.png", snapshot.image)

            if self.config.debug.save_annotated_frame:
                annotated = (
                    self._build_debug_frame(snapshot.frame_image, snapshot.bbox, inspection)
                    if snapshot.frame_image is not None
                    else self._build_debug_crop(snapshot.image, inspection)
                )
                self._write_debug_image(output_dir / f"{file_stem}__frame.png", annotated)
        except Exception:
            LOGGER.exception("debug image save failed event_id=%s output_dir=%s", inspection.event_id, output_dir)

    def _build_debug_file_stem(self, inspection: FinalizedInspection) -> str:
        return "__".join(
            [
                self._sanitize_debug_token(inspection.event_id),
                self._sanitize_debug_token(inspection.label),
                self._sanitize_debug_token(inspection.decision.decision),
            ]
        )

    def _build_debug_frame(self, image, bbox, inspection: FinalizedInspection):
        import cv2

        canvas = image.copy()
        zone_x1, zone_y1, zone_x2, zone_y2 = zone_bounds_to_pixels(
            frame_width=inspection.frame_width,
            frame_height=inspection.frame_height,
            zone=self.config.evaluation_zone,
        )
        cv2.rectangle(canvas, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 200, 0), 2)
        cv2.putText(
            canvas,
            "Evaluation Zone",
            (zone_x1, max(20, zone_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

        x1, y1, x2, y2 = bbox.to_int_tuple()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"{inspection.label} {inspection.decision.decision} {inspection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            inspection.event_id,
            (20, max(40, inspection.frame_height - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return canvas

    def _build_debug_crop(self, crop, inspection: FinalizedInspection):
        import cv2

        canvas = crop.copy()
        height, width = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (max(0, width - 1), max(0, height - 1)), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"{inspection.label} {inspection.decision.decision} {inspection.confidence:.2f}",
            (8, min(max(20, height - 8), height - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        return canvas

    def _write_debug_image(self, path: Path, image) -> None:
        import cv2

        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to write image: {path}")

    def _sanitize_debug_token(self, value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
        return token.strip("-._") or "unknown"

    def _is_metal_label(self, label: str) -> bool:
        return label.strip().lower() == "metal"

    def _supports_metal_refinement(self) -> bool:
        return not self.config.allowed_classes or any(self._is_metal_label(label) for label in self.config.allowed_classes)

    def _render_preview(self, frame) -> bool:
        import cv2

        canvas = frame.image.copy()
        zone_x1, zone_y1, zone_x2, zone_y2 = zone_bounds_to_pixels(
            frame_width=frame.width,
            frame_height=frame.height,
            zone=self.config.evaluation_zone,
        )
        cv2.rectangle(canvas, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 200, 0), 2)
        cv2.putText(
            canvas,
            "Evaluation Zone",
            (zone_x1, max(20, zone_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

        for track in self.tracks.active_tracks:
            if track.latest_snapshot is None:
                continue
            x1, y1, x2, y2 = track.latest_snapshot.bbox.to_int_tuple()
            if track.state == "emitted":
                color = (255, 0, 255)
            elif track.state == "evaluated":
                color = (0, 255, 255)
            elif track.in_evaluation_zone:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            zone_text = " in-zone" if track.in_evaluation_zone else ""
            label = f"{track.object_id} {track.state}{zone_text} z={track.in_zone_consecutive_hits} {track.confidence:.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(
            canvas,
            f"pending_events={len(self.pending_events)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Recycle Edge Runtime", canvas)
        return (cv2.waitKey(1) & 0xFF) == ord("q")
