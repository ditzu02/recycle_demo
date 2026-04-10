from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from brain.database.repository import BrainRepository
from brain.models.schema import CANONICAL_EVENT_TYPE, CANONICAL_SCHEMA_VERSION, parse_inference_payload


CLASS_LABELS = ["Metal", "Plastic", "Glass", "Paper_Cardboard", "Organic", "Other"]


def seed_repository_if_empty(repository: BrainRepository, seed: int = 20260322) -> None:
    if repository.count_events() > 0:
        return

    randomizer = random.Random(seed)
    base_time = datetime.now(UTC) - timedelta(hours=6)
    for index in range(18):
        event_time = base_time + timedelta(minutes=index * 14)
        payload = generate_mock_event(
            randomizer=randomizer,
            device_id=f"pi_{(index % 3) + 1:02d}",
            timestamp=event_time,
            sequence=index,
        )
        repository.insert_event(parse_inference_payload(payload))


def generate_mock_event(
    randomizer: random.Random,
    device_id: str,
    timestamp: datetime,
    sequence: int = 0,
) -> dict[str, Any]:
    object_count = randomizer.randint(1, 3)
    objects = []
    for object_index in range(object_count):
        label = randomizer.choice(CLASS_LABELS)
        confidence = round(randomizer.uniform(0.72, 0.98), 2)
        bbox = _random_bbox(randomizer)
        obj = {
            "object_id": f"{device_id}-{sequence:04d}-{object_index:02d}",
            "class_id": CLASS_LABELS.index(label),
            "label": label,
            "confidence": confidence,
            "bbox": bbox,
            "score": None,
            "decision": None,
            "contamination_status": None,
            "dirty_probability": None,
            "clean_probability": None,
        }
        if label == "Metal":
            dirty_probability = round(randomizer.uniform(0.05, 0.92), 2)
            contamination_status, score, decision = _metal_policy(dirty_probability)
            obj.update(
                {
                    "dirty_probability": dirty_probability,
                    "clean_probability": round(1 - dirty_probability, 2),
                    "score": score,
                    "decision": decision,
                    "contamination_status": contamination_status,
                    "refinement": {
                        "applied": True,
                        "probabilities": {
                            "dirty": dirty_probability,
                            "clean": round(1 - dirty_probability, 2),
                        },
                    },
                }
            )
        else:
            score = randomizer.choice([55, 68, 82, 90])
            decision = _generic_decision(score)
            obj.update(
                {
                    "score": score,
                    "decision": decision,
                }
            )
        objects.append(obj)

    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "event_type": CANONICAL_EVENT_TYPE,
        "event_id": f"{device_id}-{sequence:04d}",
        "device_id": device_id,
        "timestamp": timestamp.isoformat(),
        "source": {
            "type": "mock_pi",
            "index": int(device_id.split("_")[-1]),
        },
        "frame": {
            "width": 1280,
            "height": 720,
            "frame_index": sequence,
        },
        "inspection_outcome": {},
        "objects": objects,
    }


def _metal_policy(dirty_probability: float) -> tuple[str, int, str]:
    if dirty_probability >= 0.7:
        return "DIRTY", 30, "Reject"
    if dirty_probability >= 0.4:
        return "UNCERTAIN", 70, "Review"
    return "CLEAN", 95, "Accept"


def _generic_decision(score: int) -> str:
    if score >= 80:
        return "Accept"
    if score >= 50:
        return "Review"
    return "Reject"


def _random_bbox(randomizer: random.Random) -> list[int]:
    x1 = randomizer.randint(20, 420)
    y1 = randomizer.randint(20, 260)
    width = randomizer.randint(80, 220)
    height = randomizer.randint(80, 220)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x1 + width,
        "y2": y1 + height,
    }
