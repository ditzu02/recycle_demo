from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms

from edge.types import BBox, ContaminationResult, extract_snapshot_crop


def pick_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cnn(*, num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def extract_evaluation_crop(image, bbox: BBox, *, pad_ratio: float = 0.05):
    return extract_snapshot_crop(image, bbox, pad_ratio=pad_ratio)


def convert_bgr_to_rgb(image):
    return image[..., ::-1].copy()


class MetalContaminationEvaluator:
    def __init__(self, *, weights_path: str, device: torch.device | None = None, image_size: int = 224) -> None:
        self.device = device or pick_torch_device()
        self.model = build_cnn(num_classes=2).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def evaluate(self, image, bbox: BBox) -> ContaminationResult:
        crop = extract_evaluation_crop(image, bbox)
        return self.evaluate_crop(crop)

    def evaluate_crop(self, crop) -> ContaminationResult:
        if crop is None or crop.size == 0:
            return ContaminationResult(applied=False, reason="empty_crop")

        inputs = self.preprocess(convert_bgr_to_rgb(crop)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return ContaminationResult(
            dirty_probability=float(probabilities[1]),
            clean_probability=float(probabilities[0]),
            applied=True,
        )
