import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "best8S.pt"
CNN_WEIGHTS  = "metal_contamination_cnn_best.pt"

IMG_SIZE = 224
YOLO_IMGSZ = 640
YOLO_CONF  = 0.70

METAL_CLASS_NAMES = ["Metal", "metal"]


def pick_device():
    # Torch device for CNN
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda:0"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def build_cnn(num_classes=2):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def main():
    torch_device, yolo_device = pick_device()
    print("CNN device:", torch_device)
    print("YOLO device:", yolo_device)

    # Load YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    # Load CNN
    cnn = build_cnn(num_classes=2).to(torch_device)
    state = torch.load(CNN_WEIGHTS, map_location=torch_device)
    cnn.load_state_dict(state)
    cnn.eval()

    # Same normalization as training
    pre = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1) or (2).")

    # Get YOLO class names (important for metal filtering)
    names = yolo.model.names if hasattr(yolo.model, "names") else None
    print("YOLO classes:", names)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = yolo.predict(
            source=frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            device=yolo_device,
            verbose=False
        )

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Class name
                cls_name = str(cls_id)
                if names is not None and cls_id in names:
                    cls_name = names[cls_id]

                # Only run CNN if this detection is metal
                if cls_name not in METAL_CLASS_NAMES:
                    # Draw YOLO-only boxes for other materials (optional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    continue

                # Add a bit of padding like YOLO crops usually have
                pad = int(0.05 * max(x2 - x1, y2 - y1))
                xa = max(0, x1 - pad)
                ya = max(0, y1 - pad)
                xb = min(frame.shape[1], x2 + pad)
                yb = min(frame.shape[0], y2 + pad)

                crop = frame[ya:yb, xa:xb]
                if crop.size == 0:
                    continue

                # CNN prediction (Clean vs Dirty)
                inp = pre(crop).unsqueeze(0).to(torch_device)
                with torch.no_grad():
                    logits = cnn(inp)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    p_clean = float(probs[0])
                    p_dirty = float(probs[1])

                # Simple, explainable scoring tiers for metal
                if p_dirty >= 0.7:
                    score = 30
                    status = "DIRTY"
                elif p_dirty >= 0.4:
                    score = 70
                    status = "UNCERTAIN"
                else:
                    score = 95
                    status = "CLEAN"

                # Draw overlay
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if status == "DIRTY" else (255, 255, 0), 2)
                label = f"Metal {status} pD={p_dirty:.2f} score={score}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Recycle Demo (YOLO + Metal CNN)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()