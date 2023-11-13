from flask import Flask, request, jsonify
from flask_cors import CORS
from PneumoniaClassifier import VGG16

app = Flask(__name__)
CORS(app)

model = VGG16()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    base64_str = data['xray']
    
    
    prediction = model.prediction(base64_str)

    response = {
        'prediction': prediction
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
