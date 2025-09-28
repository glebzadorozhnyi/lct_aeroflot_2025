import logging
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Masks

logging.basicConfig(level=logging.DEBUG)
model = YOLO("models/yolo11m-seg.pt")

# def process_by_sam2(img):
#     predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
#     # image = Image.open(path).convert("RGB")
#     w, h = img.size
#     input_point = np.array([[w // 2, h // 2]])
#     input_label = np.array([1])
#     input_box = np.array([w // 4, h // 4, w // 4 + w // 2, h // 4 + h // 2])

#     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#         predictor.set_image(np.array(img))
#         masks, scores, logits = predictor.predict(
#             point_coords=None,  # input_point,
#             point_labels=input_label,
#             box=input_box,
#             multimask_output=False,
#         )
#     best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
#     chosen_mask = masks[best_idx].astype(np.uint8)

#     return chosen_mask


def process_by_yolo_seg(img):
    results = model.predict(img, verbose=False, conf=0.2, device="cpu")
    logging.info(f"Detected {len(results)} objects")
    return results


def pad2square(pil_image: Image.Image, pad_value: int = 0):
    image = np.array(pil_image)
    h, w = image.shape[:2]
    if h == w:
        return pil_image
    size = max(h, w)
    top = (size - h) // 2
    left = (size - w) // 2
    image = cv2.copyMakeBorder(
        image,
        top,
        size - h - top,
        left,
        size - w - left,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    return Image.fromarray(image)


def preprocess_for_clf(img: Image.Image, boxes: Boxes, segments: Masks):
    padded_image = pad2square(img)

    instruments = []
    # crop every object from image
    for box in boxes:
        instrument = padded_image.crop(box.xyxy.tolist()[0])
        instrument_resized = instrument.resize((112, 112))
        instruments.append(instrument_resized)
    # instrument.save("results.jpg")
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
        probs.append(p)

    logging.info(f"Probs from classifier: {probs}")
    return probs


def pipeline(img: Image):
    results = process_by_yolo_seg(img)

    segments = results[0].masks
    boxes = results[0].boxes
    annotated_image = results[0].plot()
    Image.fromarray(annotated_image).save("anno_image1.jpg")

    objects = preprocess_for_clf(img, boxes, segments)
    probs = process_by_clf(objects)
    return probs


if __name__ == "__main__":
    # img = cv2.imread("image.jpg")
    img = Image.open("image.jpg")
    probs = pipeline(img)
