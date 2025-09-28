from flask import Flask, render_template, Response, send_from_directory
from ultralytics import YOLO
import cv2
import os
import time
import ast
from collections import Counter
import numpy as np
import json
def get_classes(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    # Преобразуем строку в словарь
    classes_correspondence = ast.literal_eval(data)
    return classes_correspondence

def init_departues(classes_correspondence):
    for cls, _ in classes_correspondence.items():
        class_departue[cls] = 'No'

app = Flask(__name__)
model = YOLO("yolo11n-seg.pt")
progress_value = {"value": 0, "class_name": '', "departue": ''}
classes_correspondence = get_classes('classes.txt')
class_departue = dict()
init_departues(classes_correspondence)



def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    classes_counter = np.zeros(len(classes_correspondence))

    no_boxes = 0

    try:
        while True:
            success_tv, tv_frame = camera.read()

            results = model(tv_frame, verbose=False, conf=0.6)
            segments = results[0].plot()

            # классы
            mask = classes_counter > 0
            classes_counter[mask] -= 1

            indexes = np.array(results[0].boxes.cls.tolist(), dtype=int)
            confs = np.array(results[0].boxes.conf.tolist())
            boxes = np.array(results[0].boxes.xyxyn.tolist())
            indexes_set = np.array(list(set(results[0].boxes.cls.tolist())), dtype=int)

            for i in range (len(boxes)):
                if boxes[i][1] < 0.01:
                    class_departue[indexes[i]] = 'top специалист'
                if boxes[i][3] > 0.99:
                    class_departue[indexes[i]] = 'bot склад'
            classes_counter[indexes_set] += 2
            classes_counter[0] = 0

            progress_value['value'] = np.max(classes_counter)
            progress_value['class_name'] = classes_correspondence[np.argmax(classes_counter)]
            progress_value['departue'] = class_departue[np.argmax(classes_counter)]

            if progress_value['value'] == 0:
                no_boxes += 1
                if no_boxes > 20:
                    init_departues(classes_correspondence)
                    no_boxes = 0

            else:
                no_boxes = 0


            if progress_value['departue'] == 'No':
                progress_value['departue'] = class_departue[0]


            ret, buffer = cv2.imencode('.jpg', segments)
            frame_data = buffer.tobytes()

            global tv_frame_global
            tv_frame_global = segments.copy()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    finally:
        camera.release()



@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/')
def home():
    progress_value["value"] = 0
    return render_template('home.html')

@app.route('/progress')
def progress():
    def generate():
        while True:
            time.sleep(0.1)  # обновляем каждые полсекунды
            yield f"data: {json.dumps(progress_value)}\n\n"
    return Response(generate(), mimetype="text/event-stream")



if __name__ == '__main__':
    # Создаем папку static, если её нет
    os.makedirs('static', exist_ok=True)


    app.run(host='0.0.0.0', port=5000, debug=True)


