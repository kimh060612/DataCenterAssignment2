from flask import Flask, request, make_response, render_template, jsonify
import os
import requests
import base64
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

API_ENDPOINT = "http://DeepInfer:8080/image-classifier"

@app.route('/inference/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            b64_image = ""
            with open(image_path, "rb") as imageFile:
                b64_image = base64.b64encode(imageFile.read())
            data = {
                'b64': b64_image
            }
            response = requests.post(API_ENDPOINT, data=data)
            print(response.text)
            result = json.loads(response.text)
            return render_template('result.html', result = result)
    return render_template('index.html')

@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False, port=9000)