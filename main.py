# main.py


import hashlib
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from pipeline import Pipeline
from constants import TOOL_CLASSES
from db import AeroTool, AeroToolDelivery, db

templates = Jinja2Templates(directory="templates")
pipeline = Pipeline()


WORK_DIR = Path(".workdir")
HANDLED_IMAGES_DIR = Path(WORK_DIR / "handled_images")
RAW_IMAGES_DIR = Path(WORK_DIR / "raw_images")

HANDLED_IMAGES_DIR.mkdir(exist_ok=True)
RAW_IMAGES_DIR.mkdir(exist_ok=True)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=HANDLED_IMAGES_DIR), name="results")


def get_all_proc_image_names():
    return [img.name for img in HANDLED_IMAGES_DIR.iterdir()]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    cart_items = [
        {"id": 1, "name": "Плоская отвёртка"},
        {"id": 2, "name": "Крестовая отвёртка"},
        {"id": 3, "name": "Отвёртка крест"},
        {"id": 4, "name": "Коловорот"},
        {"id": 5, "name": "Пассатижи контр"},
        {"id": 6, "name": "Пассатижи"},
        {"id": 7, "name": "Шерница"},
        {"id": 8, "name": "Разводной ключ"},
        {"id": 9, "name": "Открывалка"},
        {"id": 10, "name": "Ключ рожковый"},
        {"id": 11, "name": "Бокорезы"},
    ]
    return templates.TemplateResponse("home.html", {"request": request, "cart_items": cart_items})


UPLOAD_DIR = HANDLED_IMAGES_DIR
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {file_extension} not allowed. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Check file size (note: this requires reading the file)
    # For large files, you might want to handle this differently
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
        )


def process_images_alt(image_path: Path):
    image_name = image_path.name
    with open(image_path, "rb") as fd:
        img_contents = fd.read()

        img = np.frombuffer(img_contents, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(img)
        if img is None:
            return {"error": f"Не удалось прочитать изображение {image_name}"}

        probs, annotated_image = pipeline(pil_img)
        Image.fromarray(annotated_image).save(WORK_DIR / f"handled_images/{image_name}")

    return probs


def compute_file_hash(file_path, algorithm="md5"):
    """
    Computes the hash of a file using the specified algorithm.

    Args:
        file_path (str): The path to the file.
        algorithm (str, optional): The hashing algorithm to use (e.g., "md5", "sha1", "sha256").
                                   Defaults to "md5".

    Returns:
        str: The hexadecimal representation of the file's hash, or None if the file is not found.
    """
    try:
        # Create a hash object based on the chosen algorithm
        hasher = hashlib.new(algorithm)

        # Open the file in binary read mode
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except ValueError:
        print(f"Error: Invalid hash algorithm '{algorithm}'.")
        return None


@app.post("/upload-multiple-files-advanced/")
async def upload_multiple_files_advanced(
    files: list[UploadFile] = File(
        ...,
        description="Multiple files to upload (max 10 files)",
        max_length=10,  # Limit number of files
    ),
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # if len(files) > 10:
    # raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

    uploaded_files = []
    RAW_IMAGES_DIR.mkdir(exist_ok=True)

    for file in files:
        try:
            # Validate file
            validate_file(file)
            tmp_file = RAW_IMAGES_DIR / "tmp_file.bin"
            if tmp_file.exists():
                os.remove(tmp_file)

            # Save file as tmp.bin
            with open(tmp_file, "wb") as f:
                f.write(await file.read())
            now = datetime.now()

            # Generate unique filename to avoid conflicts
            file_extension = Path(file.filename).suffix
            formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            file_hash = compute_file_hash(tmp_file, "md5")

            unique_filename = f"{file_hash}_{file.filename}"

            file_location = RAW_IMAGES_DIR / unique_filename
            shutil.copy2(tmp_file, file_location)

            probs = process_images_alt(image_path=file_location)

            # print(f"probs {probs}")

            uploaded_files.append(
                {
                    "original_filename": file.filename,
                    "saved_filename": unique_filename,
                    "content_type": file.content_type,
                    "size": file_location.stat().st_size,
                    "data": formatted_date_time,
                }
            )

            tool_counter: dict = Counter(probs)
            for i_tool in TOOL_CLASSES:
                print(f"Counter of {i_tool}: {int(tool_counter.get(i_tool, 0))}")

            delivery_set = AeroToolDelivery(
                image_file_id=unique_filename,
                founded_screw_flat=int(tool_counter.get("screw_flat", 0)),
                founded_screw_plus=int(tool_counter.get("screw_plus", 0)),
                founded_offset_plus_screw=int(tool_counter.get("offset_plus_screw", 0)),
                founded_kolovorot=int(tool_counter.get("kolovorot", 0)),
                founded_safety_pliers=int(tool_counter.get("safety_pliers", 0)),
                founded_pliers=int(tool_counter.get("pliers", 0)),
                founded_shernitsa=int(tool_counter.get("shernitsa", 0)),
                founded_adjustable_wrench=int(tool_counter.get("adjustable_wrench", 0)),
                founded_can_opener=int(tool_counter.get("can_opener", 0)),
                founded_open_end_wrench=int(tool_counter.get("open_end_wrench", 0)),
                founded_side_cutters=int(tool_counter.get("side_cutters", 0)),
                datatime=formatted_date_time,
            )
            db.add(delivery_set)  # добавляем в бд
            db.commit()

        except HTTPException as e:
            # Re-raise validation errors
            raise e
        except Exception as e:
            # Handle other errors (disk full, permissions, etc.)
            raise HTTPException(
                status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}"
            )

    return JSONResponse(
        content={
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
        }
    )


@app.post("/process-image")
async def process_images(request: Request, images: list[UploadFile]):
    image_names = []
    for image in images:
        image_name = image.filename
        img_contents = image.file.read()
        img = np.frombuffer(img_contents, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(img)
        pil_img.save(WORK_DIR / f"raw_images/{image_name}")
        if img is None:
            return {"error": f"Не удалось прочитать изображение {image_name}"}

        probs, annotated_image = pipeline.pipeline(pil_img, "yolo")
        Image.fromarray(annotated_image).save(WORK_DIR / f"/handled_images/{image_name}")

        image_names.append(image_name)

    return templates.TemplateResponse(
        name="home.html",
        context={
            "request": request,
            "first_image": image_names[0],
            "images": image_names,
        },
    )


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.get("/get_json_report")
def get_get_json_report():
    report = db.query(AeroToolDelivery).all()
    if not report:
        return JSONResponse(status_code=404, content={"message": "База пуста"})
    return report


@app.get("/get_all_uploaded_files")
def get_get_json_report():
    # list_of_uploaded_files = db.query(AeroToolDelivery).all()
    list_of_uploaded_files = [user.image_file_id for user in db.query(AeroToolDelivery).all()]
    print(list_of_uploaded_files)
    if not list_of_uploaded_files:
        return JSONResponse(status_code=404, content={"message": "База пуста"})
    return list_of_uploaded_files


@app.get("/api/get_state_for_delivery_by_id/{delivery_id}")
def get_state_for_delivery_1(delivery_id: int):
    report = db.query(AeroToolDelivery).filter(AeroToolDelivery.delivery_id == delivery_id).all()
    if not report:
        return JSONResponse(status_code=404, content={"message": "База пуста"})
    return report

@app.get("/api/get_state_for_delivery_by_name/{delivery_id_file_name}")
def get_state_for_delivery_1(delivery_id_file_name):
    report = db.query(AeroToolDelivery).filter(AeroToolDelivery.image_file_id == delivery_id_file_name).first()
    if not report:
        return JSONResponse(status_code=404, content={"message": "\"delivery_id_file_name\" not found"})
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

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, reload_excludes=".workdir/*")
