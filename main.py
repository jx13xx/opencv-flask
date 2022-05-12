from flask import Flask
from flask import jsonify, make_response, request
import cv2
import os
from PIL import Image
from detection.utils import pipeline_model

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)


@app.route('/')
def index():
    data = {
        "username": "admin",
        "email": "admin@localhost",
        "id": 42
    }
    return make_response(jsonify(data), 404)


@app.route('/blog/<int:blogID>')
def blog(blogID):
    return "This the blog id {}".format(blogID)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)

        gender = pipeline_model(path,filename, color='bgr')

        data = {
            "fileName": f.filename,
            "gender": gender,
            "path": path,
            "message": "File has been uploaded successfully"
        }

        return make_response(jsonify(data), 200)


@app.route('/camera/on')
def switch_on_camera():
    stream_webcam()


def stream_webcam():
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):
        ret, frame = cap.read()
        thresh1 = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        cv2.imshow('frame', thresh1)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
    # stream_webcam()
