import logging
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from ultralytics import SAM, YOLO
from ultralytics.engine.results import Boxes, Masks

logging.basicConfig(level=logging.DEBUG)


# def process_by_sam2(img: Image.Image):
#     sam2_checkpoint = "./models/sam2.1_hiera_tiny.pt"
#     model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

#     sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")

#     mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

#     with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
#         masks = mask_generator.generate(np.array(img))

#     return masks


def process_by_yolo(img: Image.Image):
    model = YOLO("models/best.pt")
    results = model.predict(img, verbose=False, conf=0.5, device="cpu")
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
    model_path = "models/clf.onnx"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    probs = []
    for instrument in instruments:
        p = session.run(["p"], {input_name: np.array(instrument)[None, ...]})[0][0]
        probs.append(p.tolist())

    logging.info(f"Probs from classifier: {probs}")
    return probs


def pipeline(img: Image.Image, type="yolo"):
    processing_types = {
        "yolo": process_by_yolo,
        "yolo_seg": process_by_yolo_seg,
        "sam2_ultra": process_by_sam2_ultra,
        "fast_sam": process_by_fast_sam,
    }

    results = processing_types[type](img)
    boxes = results[0].boxes

    annotated_image = results[0].plot()

    objects = preprocess_for_clf(img, boxes)
    probs = process_by_clf(objects)
    return probs, annotated_image


if __name__ == "__main__":
    img_path = "test_img_1.jpg"
    img = Image.open(img_path)
    probs, annotated_image = pipeline(img)
    Image.fromarray(annotated_image).save(img_path.split(".")[0] + "proc.jpg")
