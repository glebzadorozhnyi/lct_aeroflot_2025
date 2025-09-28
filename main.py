# main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine

SQLALCHEMY_DATABASE_URL = "sqlite:///./.workdir/sql_app.db"
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import sessionmaker

import cv2
import ast
import numpy as np
from ultralytics import YOLO
import pipeline

templates = Jinja2Templates(directory="templates")

import anyio


def read_classes(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    # Преобразуем строку в словарь
    classes_correspondence = ast.literal_eval(data)
    return classes_correspondence


model = YOLO("yolo11n-seg.pt")
templates = Jinja2Templates(directory="templates")
classes_correspondence = read_classes("classes.txt")

from starlette._utils import AwaitableOrContextManager, AwaitableOrContextManagerWrapper
from starlette.datastructures import URL, Address, FormData, Headers, QueryParams, State
from starlette.exceptions import HTTPException
from starlette.formparsers import FormParser, MultiPartException, MultiPartParser
from starlette.types import Message, Receive, Scope, Send

# создание движка
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

app = FastAPI()


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


progress_value = {"value": 0, "class_name": "", "departue": ""}


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


app.mount("/static", StaticFiles(directory="static"), name="static")


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
    success, buffer = cv2.imencode(".jpg", image_with_segments)
    if not success:
        return {"error": "Не удалось закодировать изображение"}

    # Отправляем обратно в веб
    return JSONResponse({"classes": found_classes, "image": buffer.tobytes().hex()})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


class Base(DeclarativeBase):
    pass


class AeroTool(Base):
    __tablename__ = "aerotool_set"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    delivery_state = Column(String)  # "in_stock", "on_hands"
    type = Column(
        String,
    )
    delivery_id = Column(
        Integer,
    )  # Наборов для выдачи может быть много и каждый инструмент принадлежит к одной из выдач
    detect_state = Column(
        Boolean,
    )


# создаем таблицы
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autoflush=False, bind=engine)
db = SessionLocal()


def check_exist_tool_in_db_by_name(name) -> bool:
    first = db.query(AeroTool).filter(AeroTool.name == name).first()

    if first:
        print(
            f"Find existing item with name {first.name}: id {first.id} - ({first.type}) - {first.delivery_id}"
        )
        return True
    else:
        return False


def fill_test_data():
    deliveries = [1, 2, 3]
    tool_types = [
        "screw_flat",  # 1. Плоская отвертка (-)
        "screw_plus",  # 2. Крестовая отвертка (+)
        "offset_plus_screw",  # 3. отвертка на смещенный крест
        "kolovorot",  # 4. Коловорот
        "safety_pliers",  # 5. Пассатижи контровочные
        "pliers",  # 6. Пассатижи
        "shernitsa",  # 7. Шерница
        "adjustable_wrench",  # 8. Разводной ключ
        "can_opener",  # 9. Открывалка для банок с маслом
        "open_end_wrench",  # 10. Ключ рожковый накидной 3/4
        "side_cutters",  # 11. Бокорезы
    ]
    for i_delevery in deliveries:
        for i_tool_type in tool_types:
            new_tool_unique_name = f"{i_tool_type}_{i_delevery}"
            if not check_exist_tool_in_db_by_name(new_tool_unique_name):
                screw_plus = AeroTool(
                    name=new_tool_unique_name,
                    type=i_tool_type,
                    delivery_id=i_delevery,
                    delivery_state="on_hands",
                )
                db.add(screw_plus)  # добавляем в бд
    db.commit()  # сохраняем изменения


def print_all_from_db():
    # получение всех объектов
    aero_tools = db.query(AeroTool).all()
    print(f"| id|           type        |         name         |")
    for i_aero_tool in aero_tools:
        print(f"|{i_aero_tool.id:3}| {i_aero_tool.name:22}| {i_aero_tool.type:20} |")


fill_test_data()
# print_all_from_db()


@app.get("/get_json_report")
def get_get_json_report():
    report = db.query(AeroTool).all()
    if report == None:
        return JSONResponse(status_code=404, content={"message": "База пуста"})
    return report


@app.get("/api/get_state_for_delivery/{delivery_id}")
def get_state_for_delivery_1(delivery_id: int):
    report = db.query(AeroTool).filter(AeroTool.delivery_id == delivery_id).all()
    if report == None:
        return JSONResponse(status_code=404, content={"message": "База пуста"})
    return report


@app.post("/api/set_detect_state/{aero_tool_id}")
def set_detect_state_by_id(aero_tool_id: int):
    aero_tool = db.query(AeroTool).filter(AeroTool.id == aero_tool_id).first()
    aero_tool.detect_state = True
    db.commit()  # сохраняем изменения
    db.refresh(aero_tool)
    return aero_tool


@app.post("/api/unset_detect_state/{aero_tool_id}")
def unset_detect_state_by_id(aero_tool_id: int):
    aero_tool = db.query(AeroTool).filter(AeroTool.id == aero_tool_id).first()
    aero_tool.detect_state = False
    db.commit()  # сохраняем изменения
    db.refresh(aero_tool)
    return aero_tool
