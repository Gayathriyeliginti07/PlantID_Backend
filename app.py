from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from utils.predict import predict_leaf, predict_bark

app = Flask(__name__)
CORS(app)

@app.route("/predict_leaf", methods=["POST"])
def predict_leaf_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        prediction = predict_leaf(image)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_bark", methods=["POST"])
def predict_bark_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        prediction = predict_bark(image)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)