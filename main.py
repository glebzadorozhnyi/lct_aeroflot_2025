# main.py
import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from ultralytics import YOLO

import pipeline
from db import db, AeroTool

model = YOLO("yolo11n-seg.pt")
templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    pil_img = Image.fromarray(img)
    if img is None:
        return {"error": "Не удалось прочитать изображение"}

    # Процессинг сегментатором
    probs, annotated_image = pipeline.pipeline(pil_img, "yolo")

    # Конвертируем изображение обратно в JPEG
    success, buffer = cv2.imencode(".jpg", annotated_image)
    if not success:
        return {"error": "Не удалось закодировать изображение"}

    # Отправляем обратно в веб
    return JSONResponse({"classes": probs, "image": buffer.tobytes().hex()})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.get("/get_json_report")
def get_get_json_report():
    report = db.query(AeroTool).all()
    if not report:
        return JSONResponse(status_code=404, content={"message": "База пуста"})
    return report


@app.get("/api/get_state_for_delivery/{delivery_id}")
def get_state_for_delivery_1(delivery_id: int):
    report = db.query(AeroTool).filter(AeroTool.delivery_id == delivery_id).all()
    if not report:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
