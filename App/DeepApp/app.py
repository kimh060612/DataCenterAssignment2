from tensorflow import keras as tfk
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from flask import Flask, request, make_response, render_template, jsonify
import os
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

model = ResNet50(weights='imagenet')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'static')

@app.route("/image-classifier",methods=["GET","POST"])
def classify_image():
    if request.method == "POST":
        #read and upload resized files to folder
        #json = request.get_json()
        #image = json['instances']
        #image = np.array(image)
        #image = img_to_array(image)
        img = img_to_array(load_img(BytesIO(base64.b64decode(request.form['b64'])),target_size=(224, 224)))
        # this line is added because of a bug in tf_serving(1.10.0-dev)
        img = img.astype('float16')
        image = np.expand_dims(img, axis=0)
        image = preprocess_input(image)
        prediction = model.predict(image)
        prediction = decode_predictions(prediction, top=1)[0]
        result = {}
        result['class_name'] = prediction[0][1]
        result['prob'] = float(prediction[0][2])   
        return jsonify(result)

@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, threaded=False, port=8080)


"""img_path = 'elephant.jpeg'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# 결과를 튜플의 리스트(클래스, 설명, 확률)로 디코딩합니다
# (배치 내 각 샘플 당 하나의 리스트)
print('Predicted:', decode_predictions(preds, top=1)[0])
"""