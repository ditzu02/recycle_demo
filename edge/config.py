from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: float | None = None


@dataclass
class ModelConfig:
    yolo_model_path: Path = REPO_ROOT / "best8S.pt"
    contamination_model_path: Path = REPO_ROOT / "metal_contamination_cnn_best.pt"
    yolo_image_size: int = 640


@dataclass
class ThresholdConfig:
    confidence: float = 0.70
    iou_match: float = 0.30
    stable_after_frames: int = 3
    max_missed_frames: int = 5
    dirty_review_threshold: float = 0.40
    dirty_reject_threshold: float = 0.70
    min_size_ratio: float | None = None


@dataclass
class InspectionZoneConfig:
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class EdgeConfig:
    device_id: str = "edge_demo_01"
    brain_base_url: str = "http://127.0.0.1:8000"
    event_endpoint_url: str = "http://127.0.0.1:8000/api/inference"
    heartbeat_endpoint_url: str = "http://127.0.0.1:8000/api/heartbeat"
    source_type: str = "edge_node"
    source_index: int = 0
    heartbeat_interval_seconds: float = 15.0
    request_timeout_seconds: float = 5.0
    event_retry_attempts: int = 2
    event_retry_backoff_seconds: float = 0.5
    pending_retry_delay_seconds: float = 2.0
    allowed_classes: tuple[str, ...] = ("Metal",)
    show_preview: bool = False
    camera: CameraConfig = field(default_factory=CameraConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    inspection_zone: InspectionZoneConfig = field(default_factory=InspectionZoneConfig)

    @classmethod
    def from_env(cls) -> EdgeConfig:
        brain_base_url = os.getenv("EDGE_BRAIN_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        event_endpoint = _resolve_endpoint(
            base_url=brain_base_url,
            value=os.getenv("EDGE_EVENT_ENDPOINT_URL", "/api/inference"),
        )
        heartbeat_endpoint = _resolve_endpoint(
            base_url=brain_base_url,
            value=os.getenv("EDGE_HEARTBEAT_ENDPOINT_URL", "/api/heartbeat"),
        )

        return cls(
            device_id=os.getenv("EDGE_DEVICE_ID", "edge_demo_01"),
            brain_base_url=brain_base_url,
            event_endpoint_url=event_endpoint,
            heartbeat_endpoint_url=heartbeat_endpoint,
            source_type=os.getenv("EDGE_SOURCE_TYPE", "edge_node"),
            source_index=_env_int("EDGE_SOURCE_INDEX", 0),
            heartbeat_interval_seconds=_env_float("EDGE_HEARTBEAT_INTERVAL", 15.0),
            request_timeout_seconds=_env_float("EDGE_REQUEST_TIMEOUT", 5.0),
            event_retry_attempts=_env_int("EDGE_EVENT_RETRY_ATTEMPTS", 2),
            event_retry_backoff_seconds=_env_float("EDGE_EVENT_RETRY_BACKOFF", 0.5),
            pending_retry_delay_seconds=_env_float("EDGE_PENDING_RETRY_DELAY", 2.0),
            allowed_classes=_env_csv("EDGE_ALLOWED_CLASSES", ("Metal",)),
            show_preview=_env_bool("EDGE_SHOW_PREVIEW", False),
            camera=CameraConfig(
                index=_env_int("EDGE_CAMERA_INDEX", 0),
                width=_env_int("EDGE_CAMERA_WIDTH", 1280),
                height=_env_int("EDGE_CAMERA_HEIGHT", 720),
                fps=_env_optional_float("EDGE_CAMERA_FPS"),
            ),
            models=ModelConfig(
                yolo_model_path=Path(os.getenv("EDGE_YOLO_MODEL_PATH", str(REPO_ROOT / "best8S.pt"))),
                contamination_model_path=Path(
                    os.getenv("EDGE_CONTAMINATION_MODEL_PATH", str(REPO_ROOT / "metal_contamination_cnn_best.pt"))
                ),
                yolo_image_size=_env_int("EDGE_YOLO_IMGSZ", 640),
            ),
            thresholds=ThresholdConfig(
                confidence=_env_float("EDGE_CONFIDENCE_THRESHOLD", 0.70),
                iou_match=_env_float("EDGE_IOU_THRESHOLD", 0.30),
                stable_after_frames=_env_int("EDGE_STABLE_AFTER_FRAMES", 3),
                max_missed_frames=_env_int("EDGE_MAX_MISSED_FRAMES", 5),
                dirty_review_threshold=_env_float("EDGE_DIRTY_REVIEW_THRESHOLD", 0.40),
                dirty_reject_threshold=_env_float("EDGE_DIRTY_REJECT_THRESHOLD", 0.70),
                min_size_ratio=_env_optional_float("EDGE_MIN_SIZE_RATIO"),
            ),
            inspection_zone=_parse_zone(
                os.getenv("EDGE_INSPECTION_ZONE", "0.0,0.0,1.0,1.0"),
            ),
        )

    def validate(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id must not be empty.")
        if self.thresholds.stable_after_frames < 1:
            raise ValueError("stable_after_frames must be at least 1.")
        if self.thresholds.max_missed_frames < 1:
            raise ValueError("max_missed_frames must be at least 1.")
        if self.thresholds.dirty_review_threshold > self.thresholds.dirty_reject_threshold:
            raise ValueError("dirty review threshold must be less than or equal to dirty reject threshold.")
        zone = self.inspection_zone
        if not (0.0 <= zone.x1 < zone.x2 <= 1.0 and 0.0 <= zone.y1 < zone.y2 <= 1.0):
            raise ValueError("inspection zone must be normalized and ordered within [0, 1].")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the recycle demo edge runtime.")
    parser.add_argument("--device-id")
    parser.add_argument("--brain-base-url")
    parser.add_argument("--event-endpoint-url")
    parser.add_argument("--heartbeat-endpoint-url")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-fps", type=float)
    parser.add_argument("--heartbeat-interval", type=float)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--iou-threshold", type=float)
    parser.add_argument("--stable-frames", type=int)
    parser.add_argument("--missed-frames", type=int)
    parser.add_argument("--min-size-ratio", type=float)
    parser.add_argument("--allowed-classes")
    parser.add_argument("--inspection-zone", help="Normalized x1,y1,x2,y2 rectangle.")
    parser.add_argument("--yolo-model-path")
    parser.add_argument("--contamination-model-path")
    parser.add_argument("--source-type")
    parser.add_argument("--source-index", type=int)
    parser.add_argument("--show", action="store_true")
    return parser


def build_config(argv: list[str] | None = None) -> EdgeConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = EdgeConfig.from_env()

    if args.device_id:
        config.device_id = args.device_id
    if args.brain_base_url:
        config.brain_base_url = args.brain_base_url.rstrip("/")
        if not args.event_endpoint_url:
            config.event_endpoint_url = _resolve_endpoint(config.brain_base_url, "/api/inference")
        if not args.heartbeat_endpoint_url:
            config.heartbeat_endpoint_url = _resolve_endpoint(config.brain_base_url, "/api/heartbeat")
    if args.event_endpoint_url:
        config.event_endpoint_url = _resolve_endpoint(config.brain_base_url, args.event_endpoint_url)
    if args.heartbeat_endpoint_url:
        config.heartbeat_endpoint_url = _resolve_endpoint(config.brain_base_url, args.heartbeat_endpoint_url)
    if args.camera_index is not None:
        config.camera.index = args.camera_index
    if args.camera_width is not None:
        config.camera.width = args.camera_width
    if args.camera_height is not None:
        config.camera.height = args.camera_height
    if args.camera_fps is not None:
        config.camera.fps = args.camera_fps
    if args.heartbeat_interval is not None:
        config.heartbeat_interval_seconds = args.heartbeat_interval
    if args.confidence_threshold is not None:
        config.thresholds.confidence = args.confidence_threshold
    if args.iou_threshold is not None:
        config.thresholds.iou_match = args.iou_threshold
    if args.stable_frames is not None:
        config.thresholds.stable_after_frames = args.stable_frames
    if args.missed_frames is not None:
        config.thresholds.max_missed_frames = args.missed_frames
    if args.min_size_ratio is not None:
        config.thresholds.min_size_ratio = args.min_size_ratio
    if args.allowed_classes:
        config.allowed_classes = _parse_csv(args.allowed_classes)
    if args.inspection_zone:
        config.inspection_zone = _parse_zone(args.inspection_zone)
    if args.yolo_model_path:
        config.models.yolo_model_path = Path(args.yolo_model_path)
    if args.contamination_model_path:
        config.models.contamination_model_path = Path(args.contamination_model_path)
    if args.source_type:
        config.source_type = args.source_type
    if args.source_index is not None:
        config.source_index = args.source_index
    if args.show:
        config.show_preview = True

    config.validate()
    return config


def _resolve_endpoint(base_url: str, value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return urljoin(f"{base_url.rstrip('/')}/", value.lstrip("/"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    return float(value)


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return _parse_csv(value)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_zone(value: str) -> InspectionZoneConfig:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("inspection zone must be x1,y1,x2,y2")
    return InspectionZoneConfig(*parts)
