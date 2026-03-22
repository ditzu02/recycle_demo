from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


DECISION_VALUES = {"Accept", "Review", "Reject"}
STATUS_TO_DECISION = {
    "CLEAN": "Accept",
    "UNCERTAIN": "Review",
    "DIRTY": "Reject",
}


class SchemaValidationError(ValueError):
    """Raised when an incoming inference payload is structurally invalid."""


@dataclass
class NormalizedObject:
    label: str
    confidence: float
    score: float | None = None
    decision: str | None = None
    contamination_status: str | None = None
    dirty_probability: float | None = None
    clean_probability: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    object_id: str | None = None
    class_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedEvent:
    event_uuid: str
    device_id: str
    timestamp: str
    source_type: str = "edge_node"
    source_index: int | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    frame_index: int | None = None
    objects: list[NormalizedObject] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)


def parse_inference_payload(payload: dict[str, Any]) -> NormalizedEvent:
    if not isinstance(payload, dict):
        raise SchemaValidationError("Payload must be a JSON object.")

    device_id = str(payload.get("device_id", "")).strip()
    if not device_id:
        raise SchemaValidationError("Missing required field: device_id")

    timestamp = _normalize_timestamp(payload.get("timestamp"))
    event_uuid = str(payload.get("event_id") or uuid4())

    source = payload.get("source") or {}
    frame = payload.get("frame") or {}
    object_payloads = payload.get("objects")
    if object_payloads is None:
        object_payloads = payload.get("detections", [])
    if not isinstance(object_payloads, list):
        raise SchemaValidationError("objects/detections must be an array.")

    objects = [_normalize_object(index, item) for index, item in enumerate(object_payloads)]

    return NormalizedEvent(
        event_uuid=event_uuid,
        device_id=device_id,
        timestamp=timestamp,
        source_type=str(source.get("type") or "edge_node"),
        source_index=_optional_int(source.get("index")),
        frame_width=_optional_int(frame.get("width")),
        frame_height=_optional_int(frame.get("height")),
        frame_index=_optional_int(frame.get("frame_index")),
        objects=objects,
        raw_payload=payload,
    )


def event_to_dict(event: NormalizedEvent) -> dict[str, Any]:
    data = asdict(event)
    data["objects"] = [asdict(obj) for obj in event.objects]
    return data


def _normalize_object(index: int, item: dict[str, Any]) -> NormalizedObject:
    if not isinstance(item, dict):
        raise SchemaValidationError(f"Object at index {index} must be a JSON object.")

    label = str(item.get("label") or item.get("class_label") or "").strip()
    if not label:
        raise SchemaValidationError(f"Object at index {index} is missing a label/class_label.")

    confidence = _required_float(item.get("confidence"), f"Object {index} is missing confidence.")
    bbox = _normalize_bbox(item.get("bbox"))

    refinement = item.get("refinement") or {}
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
        "refinement_applied": bool(refinement.get("applied")) if refinement else bool(label.lower() == "metal"),
    }

    return NormalizedObject(
        label=label,
        confidence=confidence,
        score=_optional_float(item.get("score")),
        decision=decision,
        contamination_status=contamination_status,
        dirty_probability=dirty_probability,
        clean_probability=clean_probability,
        bbox=bbox,
        object_id=str(item.get("object_id") or f"{index}"),
        class_id=_optional_int(item.get("class_id")),
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
