import io
from importlib.resources import files
from pathlib import Path

import make87
import numpy as np
from PIL import Image
from make87_messages.core.empty_pb2 import Empty
from make87_messages.core.header_pb2 import Header
from make87_messages.detection.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.detection.box.boxes_2d_pb2 import Boxes2DAxisAligned
from make87_messages.detection.ontology.model_ontology_pb2 import ModelOntology
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned as Box2DAxisAlignedGeometry
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
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


def main():
    make87.initialize()
    # Configuration values
    conf_threshold = make87.get_config_value("CONFIDENCE_THRESHOLD", 0.5, float)

    # Providers / Subscribers / Publishers
    ontology_endpoint = make87.get_provider("MODEL_ONTOLOGY", Empty, ModelOntology)
    jpeg_subscriber = make87.get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    detections_publisher = make87.get_publisher(name="DETECTIONS", message_type=Boxes2DAxisAligned)
    detections_endpoint = make87.get_provider(
        name="DETECTIONS",
        requester_message_type=ImageJPEG,
        provider_message_type=Boxes2DAxisAligned,
    )

    # Load model
    model_dir = Path(files("app") / "hf")
    processor, model = load_model(model_dir)

    # Ontology callback
    def ontology_callback(_: Empty) -> ModelOntology:
        header = Header()
        header.timestamp.GetCurrentTime()
        entries = [ModelOntology.ClassEntry(id=int(cid), label=lbl) for cid, lbl in model.config.id2label.items()]
        return ModelOntology(header=header, classes=entries)

    ontology_endpoint.provide(ontology_callback)

    # Detection callback
    def detections_callback(message: ImageJPEG) -> Boxes2DAxisAligned:
        # Decode JPEG to PIL Image
        image = Image.open(io.BytesIO(message.data)).convert("RGB")
        # Preprocess
        inputs = processor(images=image, return_tensors="np")
        # Inference
        ort_inputs = {k: v for k, v in inputs.items()}
        logits, boxes = model.model.run(None, ort_inputs)
        # Postprocess
        probs = softmax(logits[0])
        raw_boxes = boxes[0]
        class_ids = np.argmax(probs, axis=1)
        confidences = probs[np.arange(probs.shape[0]), class_ids]
        keep = confidences >= conf_threshold
        class_ids = class_ids[keep].astype(int)
        confidences = confidences[keep].astype(float)
        boxes_kept = raw_boxes[keep]
        boxes_xyxy = cxcywh_to_xyxy(boxes_kept, image.size).astype(float)

        # Build output message
        header = make87.header_from_message(Header, message=message, append_entity_path="hf_model")
        out = Boxes2DAxisAligned(
            header=header,
            boxes=[
                Box2DAxisAligned(
                    geometry=Box2DAxisAlignedGeometry(
                        header=header,
                        x=x0,
                        y=y0,
                        width=x1 - x0,
                        height=y1 - y0,
                    ),
                    confidence=conf,
                    class_id=cid,
                )
                for cid, conf, (x0, y0, x1, y1) in zip(class_ids, confidences, boxes_xyxy.tolist())
            ],
        )
        return out

    jpeg_subscriber.subscribe(lambda msg: detections_publisher.publish(detections_callback(msg)))
    detections_endpoint.provide(detections_callback)

    make87.loop()


if __name__ == "__main__":
    main()
