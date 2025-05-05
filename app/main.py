import io
import sys
from importlib.resources import files
from pathlib import Path

import make87
import numpy as np
import requests
from PIL import Image
from optimum.onnxruntime import ORTModel
from transformers import AutoImageProcessor


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last dimension."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def cxcywh_to_xyxy(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x0, y0, x1, y1] in pixel coordinates for all boxes."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    img_w, img_h = image_size
    x0 = (cx - w / 2) * img_w
    y0 = (cy - h / 2) * img_h
    x1 = (cx + w / 2) * img_w
    y1 = (cy + h / 2) * img_h
    return np.stack([x0, y0, x1, y1], axis=1).astype(int)


def load_model(model_dir: Path):
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
    model = ORTModel.from_pretrained(model_dir, file_name="model.onnx", provider="CPUExecutionProvider")
    return processor, model


def preprocess(image_url: str, processor):
    resp = requests.get(image_url)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    inputs = processor(images=image, return_tensors="np")
    return image, inputs


def predict(image: Image.Image, inputs: dict, model: ORTModel, conf_threshold: float = 0.5):
    ort_inputs = {k: v for k, v in inputs.items()}
    logits, boxes = model.model.run(None, ort_inputs)

    # Shape: logits (1, 300, 80), boxes (1, 300, 4)
    probs = softmax(logits[0])  # (300, 80)
    boxes = boxes[0]  # (300, 4)

    # Vectorized class/score selection
    class_ids = np.argmax(probs, axis=1)  # (300,)
    confidences = probs[np.arange(probs.shape[0]), class_ids]  # (300,)

    # Vectorized thresholding
    keep = confidences >= conf_threshold
    class_ids = class_ids[keep]
    confidences = confidences[keep]
    boxes_kept = boxes[keep]

    # Vectorized box conversion
    boxes_xyxy = cxcywh_to_xyxy(boxes_kept, image.size)  # (N, 4)

    # Combine results
    detections = list(zip(class_ids.tolist(), confidences.tolist(), map(tuple, boxes_xyxy.tolist())))
    return detections


def main():
    make87.initialize()
    model_dir = Path(files("app") / "hf")
    if not model_dir.exists():
        print(f"❌ Model directory {model_dir!r} not found.", file=sys.stderr)
        sys.exit(1)

    processor, model = load_model(model_dir)
    image_url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"

    while True:
        image, inputs = preprocess(image_url, processor)

        detections = predict(image, inputs, model)

        id2label = model.config.id2label
        print(f"✅ {len(detections)} detections with confidence ≥ 0.5")
        for class_id, conf, box in detections:
            print(f"  Class: {class_id} ({id2label[class_id]}), Confidence: {conf:.2f}, Box: {box}")


if __name__ == "__main__":
    main()
