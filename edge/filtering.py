from __future__ import annotations

from edge.config import InspectionZoneConfig
from edge.types import Detection, FrameSample


class DetectionFilter:
    def __init__(
        self,
        *,
        confidence_threshold: float,
        allowed_classes: tuple[str, ...],
        inspection_zone: InspectionZoneConfig,
        min_size_ratio: float | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = {label.lower() for label in allowed_classes}
        self.inspection_zone = inspection_zone
        self.min_size_ratio = min_size_ratio

    def filter(self, frame: FrameSample, detections: list[Detection]) -> list[Detection]:
        filtered: list[Detection] = []
        for detection in detections:
            if detection.confidence < self.confidence_threshold:
                continue
            if self.allowed_classes and detection.label.lower() not in self.allowed_classes:
                continue
            if not _center_in_zone(detection, frame.width, frame.height, self.inspection_zone):
                continue
            if self.min_size_ratio is not None and detection.bbox.area_ratio(frame.width, frame.height) < self.min_size_ratio:
                continue
            filtered.append(detection)
        return filtered


def _center_in_zone(
    detection: Detection,
    frame_width: int,
    frame_height: int,
    zone: InspectionZoneConfig,
) -> bool:
    center_x, center_y = detection.bbox.center
    min_x = zone.x1 * frame_width
    max_x = zone.x2 * frame_width
    min_y = zone.y1 * frame_height
    max_y = zone.y2 * frame_height
    return min_x <= center_x <= max_x and min_y <= center_y <= max_y
