from PIL import Image
from ultralytics import YOLO

from constants import TOOL_CLASSES, TOOL_CLASSES_RU


class Pipeline:
    _model: YOLO = None

    @classmethod
    def get_model(cls, model_path: str):
        if cls._model is None:
            cls._model = YOLO(model_path)
        return cls._model

    def __init__(self, model_path: str = "models/best.pt"):
        self.model = Pipeline.get_model(model_path)

    def process_by_yolo(self, img: Image.Image):
        results = self.model.predict(img, verbose=False, conf=0.5, device="0", iou=0.1)
        return results

    def process(self, img: Image.Image):
        results = self.process_by_yolo(img)

        locale_names = {i: tool for i, tool in enumerate(TOOL_CLASSES_RU)}
        results[0].names = locale_names
        annotated_image_yolo = results[0].plot()
        indexes = results[0].boxes.cls.tolist()
        classes = [TOOL_CLASSES[int(x)] for x in indexes]

        return classes, annotated_image_yolo

    def __call__(self, img):
        return self.process(img=img)


if __name__ == "__main__":
    img_path = "image2.jpg"
    img = Image.open(img_path)
    pipeline = Pipeline()
    probs, annotated_image = pipeline(img)
    Image.fromarray(annotated_image).save(img_path.split(".")[0] + "proc.jpg")
