from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from edge.types import FinalizedInspection


CANONICAL_SCHEMA_VERSION = "brain-v1"
CANONICAL_EVENT_TYPE = "inspection.finalized"


def build_event_id(
    *,
    device_id: str,
    frame_index: int,
    track_number: int,
    session_identifier: str | None = None,
) -> str:
    prefix = f"{device_id}-{session_identifier}" if session_identifier else device_id
    return f"{prefix}-f{frame_index:06d}-t{track_number:04d}"


def map_event_payload(inspection: FinalizedInspection) -> dict[str, Any]:
    contamination = inspection.contamination
    dirty_probability, clean_probability = _canonical_probabilities(contamination)
    object_payload: dict[str, Any] = {
        "object_id": inspection.object_id,
        "class_id": inspection.class_id,
        "label": inspection.label,
        "confidence": round(inspection.confidence, 4),
        "bbox": inspection.bbox.to_dict(),
        "score": inspection.decision.score,
        "decision": inspection.decision.decision,
        "contamination_status": inspection.decision.contamination_status,
        "dirty_probability": dirty_probability,
        "clean_probability": clean_probability,
    }
    if contamination is not None and contamination.applied and contamination.available:
        object_payload["refinement"] = {
            "applied": True,
            "probabilities": {
                "dirty": contamination.dirty_probability,
                "clean": contamination.clean_probability,
            },
        }

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "event_type": CANONICAL_EVENT_TYPE,
        "event_id": inspection.event_id,
        "device_id": inspection.device_id,
        "timestamp": _format_timestamp(inspection.timestamp),
        "source": {
            "type": inspection.source_type,
            "index": inspection.source_index,
        },
        "frame": {
            "width": inspection.frame_width,
            "height": inspection.frame_height,
            "frame_index": inspection.frame_index,
        },
        "inspection_outcome": inspection.inspection_outcome,
        "objects": [object_payload],
    }


def map_heartbeat_payload(
    *,
    device_id: str,
    timestamp: datetime | None = None,
    status: str = "online",
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "device_id": device_id,
        "timestamp": _format_timestamp(timestamp or datetime.now(UTC)),
        "status": status,
    }


def _format_timestamp(timestamp: datetime) -> str:
    if timestamp.tzinfo is None:
        normalized = timestamp.replace(tzinfo=UTC)
    else:
        normalized = timestamp.astimezone(UTC)
    return normalized.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_probabilities(contamination) -> tuple[float, float]:
    if contamination is None:
        return (0.5, 0.5)
    dirty = contamination.dirty_probability
    clean = contamination.clean_probability
    if dirty is None or clean is None:
        return (0.5, 0.5)
    return (dirty, clean)
