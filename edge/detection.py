from __future__ import annotations

from typing import Any

from edge.types import BBox, Detection, FrameSample


def pick_yolo_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLODetector:
    def __init__(
        self,
        *,
        model_path: str,
        image_size: int = 640,
        confidence_floor: float = 0.05,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.image_size = image_size
        self.confidence_floor = confidence_floor
        self.device = device or pick_yolo_device()
        raw_names = self.model.model.names if hasattr(self.model.model, "names") else {}
        self.class_names = dict(raw_names) if isinstance(raw_names, dict) else {index: name for index, name in enumerate(raw_names)}

    def detect(self, frame: FrameSample) -> list[Detection]:
        results = self.model.predict(
            source=frame.image,
            imgsz=self.image_size,
            conf=self.confidence_floor,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, classes):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            detections.append(
                Detection(
                    label=str(self.class_names.get(class_id, class_id)),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    frame_index=frame.index,
                )
            )
        return detections
