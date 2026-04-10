from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def expand(self, ratio: float, frame_width: int, frame_height: int) -> BBox:
        pad = ratio * max(self.width, self.height)
        return BBox(
            x1=max(0.0, self.x1 - pad),
            y1=max(0.0, self.y1 - pad),
            x2=min(float(frame_width), self.x2 + pad),
            y2=min(float(frame_height), self.y2 + pad),
        )

    def intersection_over_union(self, other: BBox) -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        union = self.area + other.area - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def area_ratio(self, frame_width: int, frame_height: int) -> float:
        frame_area = max(1, frame_width * frame_height)
        return self.area / float(frame_area)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class FrameSample:
    index: int
    image: np.ndarray
    captured_at: datetime
    width: int
    height: int

    @classmethod
    def from_image(
        cls,
        *,
        index: int,
        image: np.ndarray,
        captured_at: datetime | None = None,
    ) -> FrameSample:
        height, width = image.shape[:2]
        return cls(
            index=index,
            image=image,
            captured_at=captured_at or datetime.now(UTC),
            width=width,
            height=height,
        )


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox
    class_id: int | None = None
    frame_index: int | None = None


@dataclass
class ContaminationResult:
    dirty_probability: float | None = None
    clean_probability: float | None = None
    applied: bool = False
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.dirty_probability is not None and self.clean_probability is not None


@dataclass(frozen=True)
class DecisionResult:
    decision: str
    contamination_status: str
    score: int
    reason: str


@dataclass
class TrackSnapshot:
    frame_index: int
    captured_at: datetime
    frame_width: int
    frame_height: int
    bbox: BBox
    confidence: float
    label: str
    class_id: int | None
    image: np.ndarray | None = None


@dataclass
class TrackState:
    track_number: int
    object_id: str
    state: str = "tentative"
    class_id: int | None = None
    label: str = ""
    confidence: float = 0.0
    seen_frames: int = 0
    consecutive_hits: int = 0
    missed_frames: int = 0
    label_history: list[str] = field(default_factory=list)
    latest_snapshot: TrackSnapshot | None = None
    best_snapshot: TrackSnapshot | None = None
    contamination: ContaminationResult | None = None
    decision: DecisionResult | None = None
    event_id: str | None = None

    def observe(self, frame: FrameSample, detection: Detection) -> None:
        snapshot = TrackSnapshot(
            frame_index=frame.index,
            captured_at=frame.captured_at,
            frame_width=frame.width,
            frame_height=frame.height,
            bbox=detection.bbox,
            confidence=detection.confidence,
            label=detection.label,
            class_id=detection.class_id,
            image=None,
        )
        self.latest_snapshot = snapshot
        self.class_id = detection.class_id
        self.label = detection.label
        self.confidence = detection.confidence
        self.seen_frames += 1
        self.consecutive_hits += 1
        self.missed_frames = 0
        self.label_history.append(detection.label)

        if self.best_snapshot is None or detection.confidence >= self.best_snapshot.confidence:
            self.best_snapshot = TrackSnapshot(
                frame_index=frame.index,
                captured_at=frame.captured_at,
                frame_width=frame.width,
                frame_height=frame.height,
                bbox=detection.bbox,
                confidence=detection.confidence,
                label=detection.label,
                class_id=detection.class_id,
                image=frame.image.copy(),
            )

    def miss(self) -> None:
        self.missed_frames += 1
        self.consecutive_hits = 0


@dataclass(frozen=True)
class FinalizedInspection:
    event_id: str
    device_id: str
    source_type: str
    source_index: int
    timestamp: datetime
    frame_index: int
    frame_width: int
    frame_height: int
    object_id: str
    track_number: int
    class_id: int | None
    label: str
    confidence: float
    bbox: BBox
    decision: DecisionResult
    contamination: ContaminationResult | None = None
    inspection_outcome: dict[str, Any] = field(default_factory=dict)
