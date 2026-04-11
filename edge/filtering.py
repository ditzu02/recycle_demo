from __future__ import annotations

from edge.config import EvaluationZoneConfig
from edge.types import BBox, Detection, FrameSample


class DetectionFilter:
    def __init__(
        self,
        *,
        confidence_threshold: float,
        allowed_classes: tuple[str, ...],
        min_size_ratio: float | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = {label.lower() for label in allowed_classes}
        self.min_size_ratio = min_size_ratio

    def filter(self, frame: FrameSample, detections: list[Detection]) -> list[Detection]:
        filtered: list[Detection] = []
        for detection in detections:
            if detection.confidence < self.confidence_threshold:
                continue
            if self.allowed_classes and detection.label.lower() not in self.allowed_classes:
                continue
            if self.min_size_ratio is not None and detection.bbox.area_ratio(frame.width, frame.height) < self.min_size_ratio:
                continue
            filtered.append(detection)
        return filtered


def is_detection_center_in_zone(
    detection: Detection,
    frame: FrameSample,
    zone: EvaluationZoneConfig,
) -> bool:
    return is_bbox_center_in_zone(
        detection.bbox,
        frame_width=frame.width,
        frame_height=frame.height,
        zone=zone,
    )


def is_bbox_center_in_zone(
    bbox: BBox,
    *,
    frame_width: int,
    frame_height: int,
    zone: EvaluationZoneConfig,
) -> bool:
    center_x, center_y = bbox.center
    min_x = zone.x1 * frame_width
    max_x = zone.x2 * frame_width
    min_y = zone.y1 * frame_height
    max_y = zone.y2 * frame_height
    return min_x <= center_x <= max_x and min_y <= center_y <= max_y


def zone_bounds_to_pixels(
    *,
    frame_width: int,
    frame_height: int,
    zone: EvaluationZoneConfig,
) -> tuple[int, int, int, int]:
    return (
        int(zone.x1 * frame_width),
        int(zone.y1 * frame_height),
        int(zone.x2 * frame_width),
        int(zone.y2 * frame_height),
    )
