from flask import Flask, render_template, Response, send_from_directory
from ultralytics import YOLO
import cv2
import os
import time

app = Flask(__name__)
model = YOLO("yolo11n-seg.pt")  # load an official model


def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            success_tv, tv_frame = camera.read()

            results = model(tv_frame, verbose=False)
            segments = results[0].plot()

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
    return render_template('home.html')


if __name__ == '__main__':
    # Создаем папку static, если её нет
    os.makedirs('static', exist_ok=True)


    app.run(host='0.0.0.0', port=5000, debug=True)