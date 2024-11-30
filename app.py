import base64
import cv2
import numpy as np
import flask
import io
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

from low_light import low_image_enhancement

import webbrowser
from threading import Timer

app = flask.Flask(__name__)
CORS(app)

@app.route("/", methods=["Get","POST"])
@cross_origin()
def predict():

    global temp_data

    data = {"success": False}
    if request.method == "GET":
        return render_template("index.html")
    
    elif flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            npimg = np.fromstring(image, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            retval, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            enhanced_img = low_image_enhancement(img)
            retval, enh_buffer = cv2.imencode('.jpg', enhanced_img)
            enhanced_base64 = base64.b64encode(enh_buffer).decode('utf-8')

            data['success'] = True
            data['image'] = str(img_base64)
            data['success'] = True
            data['image'] = str(img_base64)

            return render_template("index.html",original_image = img_base64, enhanced_image=enhanced_base64)
        
    return jsonify(data)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

import os

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"))

    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        Timer(1, open_browser).start()

    app.run(debug=False)
