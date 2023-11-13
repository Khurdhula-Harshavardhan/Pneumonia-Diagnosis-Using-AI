from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
import numpy as np
from PneumoniaClassifier import VGG16

app = Flask(__name__)
CORS(app)

model = VGG16()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    base64_str = data['xray']
    
    
    prediction = model.predict(base64_str)

    response = {
        'prediction': prediction
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
