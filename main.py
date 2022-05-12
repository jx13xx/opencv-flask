from flask import Flask
from flask import json
from flask import jsonify, make_response, request
import cv2
import os

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
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)

        return "File Uploaded Successfully!"


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
