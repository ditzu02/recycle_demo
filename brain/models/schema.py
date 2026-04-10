from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


DECISION_VALUES = {"Accept", "Review", "Reject"}
STATUS_TO_DECISION = {
    "CLEAN": "Accept",
    "UNCERTAIN": "Review",
    "DIRTY": "Reject",
}
CANONICAL_SCHEMA_VERSION = "brain-v1"
CANONICAL_EVENT_TYPE = "inspection.finalized"


class SchemaValidationError(ValueError):
    """Raised when an incoming edge payload is structurally invalid."""


@dataclass
class NormalizedObject:
    object_id: str
    label: str
    confidence: float
    class_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    score: float | None = None
    decision: str | None = None
    contamination_status: str | None = None
    dirty_probability: float | None = None
    clean_probability: float | None = None
    refinement: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedEvent:
    event_uuid: str
    device_id: str
    timestamp: str
    schema_version: str = CANONICAL_SCHEMA_VERSION
    event_type: str = CANONICAL_EVENT_TYPE
    source_type: str = "edge_node"
    source_index: int | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    frame_index: int | None = None
    inspection_outcome: dict[str, Any] = field(default_factory=dict)
    objects: list[NormalizedObject] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedHeartbeat:
    schema_version: str
    device_id: str
    timestamp: str
    status: str
    raw_payload: dict[str, Any] = field(default_factory=dict)


def parse_inference_payload(payload: dict[str, Any]) -> NormalizedEvent:
    if not isinstance(payload, dict):
        raise SchemaValidationError("Payload must be a JSON object.")

    explicit_brain_v1 = "schema_version" in payload or "event_type" in payload
    schema_version = _normalize_schema_version(payload.get("schema_version"), explicit=explicit_brain_v1)
    event_type = _normalize_event_type(payload.get("event_type"), explicit=explicit_brain_v1)

    if explicit_brain_v1:
        _require_top_level_keys(
            payload,
            (
                "schema_version",
                "event_type",
                "event_id",
                "device_id",
                "timestamp",
                "source",
                "frame",
                "inspection_outcome",
                "objects",
            ),
        )

    device_id = str(payload.get("device_id", "")).strip()
    if not device_id:
        raise SchemaValidationError("Missing required field: device_id")

    timestamp = _normalize_timestamp(payload.get("timestamp"))
    event_id = payload.get("event_id")
    if explicit_brain_v1 and event_id in (None, ""):
        raise SchemaValidationError("Missing required field: event_id")
    event_uuid = str(event_id or uuid4())

    source = _normalize_object_container(payload.get("source"), "source", required=explicit_brain_v1)
    frame = _normalize_object_container(payload.get("frame"), "frame", required=explicit_brain_v1)
    inspection_outcome = _normalize_object_container(
        payload.get("inspection_outcome"),
        "inspection_outcome",
        required=explicit_brain_v1,
    )
    object_payloads = payload.get("objects")
    if object_payloads is None:
        object_payloads = payload.get("detections", [])
    if not isinstance(object_payloads, list):
        raise SchemaValidationError("objects/detections must be an array.")

    objects = [
        _normalize_object(index, item, require_canonical_keys=explicit_brain_v1)
        for index, item in enumerate(object_payloads)
    ]

    return NormalizedEvent(
        event_uuid=event_uuid,
        device_id=device_id,
        timestamp=timestamp,
        schema_version=schema_version,
        event_type=event_type,
        source_type=str(source.get("type") or "edge_node"),
        source_index=_optional_int(source.get("index")),
        frame_width=_optional_int(frame.get("width")),
        frame_height=_optional_int(frame.get("height")),
        frame_index=_optional_int(frame.get("frame_index")),
        inspection_outcome=inspection_outcome,
        objects=objects,
        raw_payload=payload,
    )


def event_to_dict(event: NormalizedEvent) -> dict[str, Any]:
    return {
        "schema_version": event.schema_version,
        "event_type": event.event_type,
        "event_id": event.event_uuid,
        "device_id": event.device_id,
        "timestamp": event.timestamp,
        "source": {
            "type": event.source_type,
            "index": event.source_index,
        },
        "frame": {
            "width": event.frame_width,
            "height": event.frame_height,
            "frame_index": event.frame_index,
        },
        "inspection_outcome": event.inspection_outcome,
        "objects": [object_to_dict(obj) for obj in event.objects],
    }


def heartbeat_to_dict(heartbeat: NormalizedHeartbeat) -> dict[str, Any]:
    return {
        "schema_version": heartbeat.schema_version,
        "device_id": heartbeat.device_id,
        "timestamp": heartbeat.timestamp,
        "status": heartbeat.status,
    }


def parse_heartbeat_payload(payload: dict[str, Any]) -> NormalizedHeartbeat:
    if not isinstance(payload, dict):
        raise SchemaValidationError("Payload must be a JSON object.")

    _require_top_level_keys(payload, ("schema_version", "device_id", "timestamp", "status"))
    schema_version = _normalize_schema_version(payload.get("schema_version"), explicit=True)
    device_id = str(payload.get("device_id", "")).strip()
    if not device_id:
        raise SchemaValidationError("Missing required field: device_id")

    status = str(payload.get("status", "")).strip().lower()
    if not status:
        raise SchemaValidationError("Missing required field: status")

    return NormalizedHeartbeat(
        schema_version=schema_version,
        device_id=device_id,
        timestamp=_normalize_timestamp(payload.get("timestamp")),
        status=status,
        raw_payload=payload,
    )


def object_to_dict(obj: NormalizedObject) -> dict[str, Any]:
    payload = {
        "object_id": obj.object_id,
        "class_id": obj.class_id,
        "label": obj.label,
        "confidence": obj.confidence,
        "bbox": _bbox_to_dict(obj.bbox),
        "score": obj.score,
        "decision": obj.decision,
        "contamination_status": obj.contamination_status,
        "dirty_probability": obj.dirty_probability,
        "clean_probability": obj.clean_probability,
    }
    if obj.refinement is not None:
        payload["refinement"] = obj.refinement
    return payload


def _normalize_object(index: int, item: dict[str, Any], require_canonical_keys: bool) -> NormalizedObject:
    if not isinstance(item, dict):
        raise SchemaValidationError(f"Object at index {index} must be a JSON object.")

    if require_canonical_keys:
        _require_top_level_keys(
            item,
            (
                "object_id",
                "class_id",
                "label",
                "confidence",
                "bbox",
                "score",
                "decision",
                "contamination_status",
                "dirty_probability",
                "clean_probability",
            ),
            label=f"Object at index {index}",
        )

    label = str(item.get("label") or item.get("class_label") or "").strip()
    if not label:
        raise SchemaValidationError(f"Object at index {index} is missing a label/class_label.")

    confidence = _required_float(item.get("confidence"), f"Object {index} is missing confidence.")
    bbox = _normalize_bbox(item.get("bbox"))

    raw_refinement = item.get("refinement")
    if raw_refinement is not None and not isinstance(raw_refinement, dict):
        raise SchemaValidationError(f"Object at index {index} has a non-object refinement field.")
    refinement = raw_refinement or {}
    probabilities = refinement.get("probabilities") or {}
    dirty_probability = _optional_float(item.get("dirty_probability"))
    clean_probability = _optional_float(item.get("clean_probability"))

    if dirty_probability is None:
        dirty_probability = _optional_float(probabilities.get("dirty"))
    if clean_probability is None:
        clean_probability = _optional_float(probabilities.get("clean"))

    raw_status = (
        item.get("contamination_status")
        or item.get("status")
        or item.get("decision")
    )
    contamination_status = _normalize_status(raw_status, item.get("score"), dirty_probability)
    decision = _normalize_decision(item.get("decision"), item.get("score"), dirty_probability, contamination_status)

    metadata = {
        "crop_bbox": item.get("crop_bbox"),
        "mask": item.get("mask"),
        "refinement": raw_refinement,
        "refinement_applied": bool(refinement.get("applied")) if refinement else bool(label.lower() == "metal"),
    }

    return NormalizedObject(
        object_id=str(item.get("object_id") or f"{index}"),
        label=label,
        confidence=confidence,
        class_id=_optional_int(item.get("class_id")),
        bbox=bbox,
        score=_optional_float(item.get("score")),
        decision=decision,
        contamination_status=contamination_status,
        dirty_probability=dirty_probability,
        clean_probability=clean_probability,
        refinement=raw_refinement,
        metadata=metadata,
    )


def _normalize_timestamp(value: Any) -> str:
    if value in (None, ""):
        return datetime.now(UTC).isoformat()
    if not isinstance(value, str):
        raise SchemaValidationError("timestamp must be a string in ISO 8601 format.")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SchemaValidationError("timestamp must be valid ISO 8601.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(key in value for key in keys):
            raise SchemaValidationError("bbox dict must contain x1, y1, x2, y2.")
        return tuple(float(value[key]) for key in keys)
    if isinstance(value, list | tuple) and len(value) == 4:
        return tuple(float(component) for component in value)
    raise SchemaValidationError("bbox must be [x1, y1, x2, y2] or an object with x1/y1/x2/y2.")


def _bbox_to_dict(value: tuple[float, float, float, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    x1, y1, x2, y2 = value
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _normalize_status(raw_value: Any, score: Any, dirty_probability: float | None) -> str | None:
    if raw_value is not None:
        text = str(raw_value).strip()
        upper = text.upper()
        if upper in STATUS_TO_DECISION:
            return upper
        if text.title() in DECISION_VALUES:
            return None
    numeric_score = _optional_float(score)
    if numeric_score is not None:
        if numeric_score >= 80:
            return "CLEAN"
        if numeric_score >= 50:
            return "UNCERTAIN"
        return "DIRTY"
    if dirty_probability is not None:
        if dirty_probability >= 0.7:
            return "DIRTY"
        if dirty_probability >= 0.4:
            return "UNCERTAIN"
        return "CLEAN"
    return None


def _normalize_decision(
    raw_value: Any,
    score: Any,
    dirty_probability: float | None,
    contamination_status: str | None,
) -> str | None:
    if raw_value is not None:
        text = str(raw_value).strip()
        titled = text.title()
        upper = text.upper()
        if titled in DECISION_VALUES:
            return titled
        if upper in STATUS_TO_DECISION:
            return STATUS_TO_DECISION[upper]
    if contamination_status in STATUS_TO_DECISION:
        return STATUS_TO_DECISION[contamination_status]
    numeric_score = _optional_float(score)
    if numeric_score is not None:
        if numeric_score >= 80:
            return "Accept"
        if numeric_score >= 50:
            return "Review"
        return "Reject"
    if dirty_probability is not None:
        if dirty_probability <= 0.3:
            return "Accept"
        if dirty_probability <= 0.6:
            return "Review"
        return "Reject"
    return None


def _required_float(value: Any, message: str) -> float:
    result = _optional_float(value)
    if result is None:
        raise SchemaValidationError(message)
    return result


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _normalize_schema_version(value: Any, explicit: bool) -> str:
    if value in (None, ""):
        return CANONICAL_SCHEMA_VERSION
    text = str(value).strip()
    if text != CANONICAL_SCHEMA_VERSION:
        raise SchemaValidationError(
            f"schema_version must be {CANONICAL_SCHEMA_VERSION!r}."
        )
    return text


def _normalize_event_type(value: Any, explicit: bool) -> str:
    if value in (None, ""):
        return CANONICAL_EVENT_TYPE
    text = str(value).strip()
    if text != CANONICAL_EVENT_TYPE:
        raise SchemaValidationError(
            f"event_type must be {CANONICAL_EVENT_TYPE!r}."
        )
    return text


def _normalize_object_container(value: Any, name: str, required: bool) -> dict[str, Any]:
    if value in (None, ""):
        if required:
            raise SchemaValidationError(f"Missing required field: {name}")
        return {}
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{name} must be a JSON object.")
    return value


def _require_top_level_keys(
    payload: dict[str, Any],
    required_keys: tuple[str, ...],
    label: str = "Payload",
) -> None:
    for key in required_keys:
        if key not in payload:
            raise SchemaValidationError(f"{label} is missing required field: {key}")
