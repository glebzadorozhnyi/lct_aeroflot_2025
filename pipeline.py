import logging
from typing import Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import SAM, YOLO
from ultralytics.engine.results import Boxes

from constants import TOOL_CLASSES, TOOL_CLASSES_RU

logging.basicConfig(level=logging.DEBUG)


def process_by_yolo(img: Image.Image):
    model = YOLO("models/best.pt")
    results = model.predict(img, verbose=False, conf=0.5, device="0", iou=0.1)
    logging.info(f"Detected {len(results)} objects")
    return results


def process_by_yolo_seg(img: Image.Image):
    model = YOLO("models/yolo11m-seg.pt")
    results = model.predict(img, verbose=False, conf=0.2, device="cpu")
    logging.info(f"Detected {len(results)} objects")
    return results


def process_by_sam2_ultra(img: Image.Image):
    model = SAM("models/sam2.1_t.pt")
    results = model.predict(img, verbose=False, conf=0.2, device="cpu")
    logging.info(f"Detected {len(results)} objects")
    return results


def process_by_fast_sam(img: Image.Image):
    model = SAM("models/FastSAM-s.pt.pt")
    results = model.predict(img, verbose=False, conf=0.2, device="cpu")
    logging.info(f"Detected {len(results)} objects")
    return results


def preprocess_for_clf(img: Image.Image, boxes: Boxes):
    instruments = []
    # crop every object from image
    for box in boxes:
        instrument = img.crop(box.xyxy.tolist()[0])
        instrument_resized = instrument.resize((112, 112))
        instruments.append(instrument_resized)
    # TODO: use segments to crop masked object on white background

    return instruments


def process_by_clf(instruments: Sequence[Image.Image]):
    model_path = "models/clf_screw.onnx"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    probs = []
    for instrument in instruments:
        p = session.run(["p"], {input_name: np.array(instrument)[None, ...]})[0][0]
        probs.append(p.tolist())

    return probs


def pipeline(img: Image.Image, type="yolo"):
    processing_types = {
        "yolo": process_by_yolo,
        "yolo_seg": process_by_yolo_seg,
        "sam2_ultra": process_by_sam2_ultra,
        "fast_sam": process_by_fast_sam,
    }
    results = processing_types[type](img)

    locale_names = {i: tool for i, tool in enumerate(TOOL_CLASSES_RU)}
    results[0].names = locale_names
    annotated_image_yolo = results[0].plot()
    indexes = results[0].boxes.cls.tolist()
    classes = [TOOL_CLASSES[int(x)] for x in indexes]

    return classes, annotated_image_yolo


if __name__ == "__main__":
    img_path = "test_img_1.jpg"
    img = Image.open(img_path)
    probs, annotated_image = pipeline(img)
    Image.fromarray(annotated_image).save(img_path.split(".")[0] + "proc.jpg")
