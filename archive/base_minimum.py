from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import ast
import numpy as np
from ultralytics import YOLO

def read_classes(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    # Преобразуем строку в словарь
    classes_correspondence = ast.literal_eval(data)
    return classes_correspondence

app = FastAPI()
model = YOLO("yolo11n-seg.pt")
templates = Jinja2Templates(directory="templates")
classes_correspondence = read_classes('classes.txt')

# Разрешаем запросы с любых источников (можно ограничить до вашего фронта)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # замените "*" на URL вашего фронта при необходимости
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_class_names(indexes, class_dict):
    classes = list()
    for cls_index in indexes:
        classes.append(class_dict[cls_index])
    return classes


def process_with_segmentation(image):
    """
    :param      image: картина для обработки
    :return:    segments: картинка с отрисованными сегментациями для отправки во фронт
                classes: массив с найденными классами
    """
    results = model(image, verbose=False, conf=0.2)
    segments = results[0].plot()
    indexes_set_list = list(set(results[0].boxes.cls.tolist()))
    classes = get_class_names(indexes_set_list, classes_correspondence)
    return segments, classes


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    # Читаем картинку в массив байт
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    # Декодируем изображение OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Не удалось прочитать изображение"}

    # Процессинг сегментатором
    image_with_segments, found_classes = process_with_segmentation(img)

    # Конвертируем изображение обратно в JPEG
    success, buffer = cv2.imencode('.jpg', image_with_segments)
    if not success:
        return {"error": "Не удалось закодировать изображение"}

    # Отправляем обратно в веб
    return JSONResponse(
        {
            "classes": found_classes,
            "image": buffer.tobytes().hex()
        }
    )

# Запуск через uvicorn:
#  uvicorn base_minimum:app --reload --host 0.0.0.0 --port 8000
