from __future__ import annotations

from pathlib import Path
from typing import Any

from edge.config import DEFAULT_YOLO_CONFIDENCE_FLOOR
from edge.types import BBox, Detection, FrameSample


SUPPORTED_YOLO_MODEL_FORMATS = {"pt", "onnx"}


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
        confidence_floor: float = DEFAULT_YOLO_CONFIDENCE_FLOOR,
        class_names: tuple[str, ...] | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_format = infer_yolo_model_format(self.model_path)
        if self.model_format not in SUPPORTED_YOLO_MODEL_FORMATS:
            raise ValueError(f"Unsupported YOLO model format: {self.model_path.suffix or '<none>'}")

        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.image_size = image_size
        self.confidence_floor = confidence_floor
        self.device = device or pick_yolo_device()
        model_backend = getattr(self.model, "model", None)
        raw_names = getattr(model_backend, "names", {}) or getattr(self.model, "names", {})
        self.class_names = normalize_class_names(raw_names)
        if not self.class_names and class_names:
            self.class_names = {index: name for index, name in enumerate(class_names)}

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


def infer_yolo_model_format(model_path: str | Path) -> str:
    return Path(model_path).suffix.lower().lstrip(".")


def normalize_class_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(index): str(name) for index, name in raw_names.items()}
    if isinstance(raw_names, list | tuple):
        return {index: str(name) for index, name in enumerate(raw_names)}
    return {}
