# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # from PIL import Image
# # import io
# # from utils.predict import predict_leaf, predict_bark

# # app = Flask(__name__)
# # CORS(app)

# # @app.route("/predict_leaf", methods=["POST"])
# # def predict_leaf_route():
# #     if "file" not in request.files:
# #         return jsonify({"error": "No file uploaded"}), 400

# #     file = request.files["file"]
# #     try:
# #         image = Image.open(io.BytesIO(file.read())).convert("RGB")
# #         prediction = predict_leaf(image)
# #         return jsonify(prediction)
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route("/predict_bark", methods=["POST"])
# # def predict_bark_route():
# #     if "file" not in request.files:
# #         return jsonify({"error": "No file uploaded"}), 400

# #     file = request.files["file"]
# #     try:
# #         image = Image.open(io.BytesIO(file.read())).convert("RGB")
# #         prediction = predict_bark(image)
# #         return jsonify(prediction)
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == "__main__":

# #     app.run(debug=True)

# app.py
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from utils.predict import predict_leaf, predict_bark

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plantid-backend")

app = Flask(__name__)
CORS(app)  # Allow frontend to access API

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ PlantID Backend Running Successfully"}), 200

@app.route("/predict_leaf", methods=["POST"])
def predict_leaf_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (field name must be 'file')"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        logger.exception("Failed to read uploaded image for leaf prediction")
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    try:
        prediction = predict_leaf(image)
        # prediction may already contain an "error" key from predict.py; forward that with 500
        if isinstance(prediction, dict) and "error" in prediction:
            logger.error("Leaf model returned error: %s", prediction["error"])
            return jsonify(prediction), 500
        return jsonify(prediction), 200
    except Exception as e:
        logger.exception("Leaf prediction failed")
        return jsonify({"error": f"Leaf prediction failed: {str(e)}"}), 500

@app.route("/predict_bark", methods=["POST"])
def predict_bark_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (field name must be 'file')"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        logger.exception("Failed to read uploaded image for bark prediction")
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    try:
        prediction = predict_bark(image)
        if isinstance(prediction, dict) and "error" in prediction:
            logger.error("Bark model returned error: %s", prediction["error"])
            return jsonify(prediction), 500
        return jsonify(prediction), 200
    except Exception as e:
        logger.exception("Bark prediction failed")
        return jsonify({"error": f"Bark prediction failed: {str(e)}"}), 500

# Useful health-check route for readiness
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Render and other PaaS provide PORT in env vars. Default to 5000 for local runs.
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting PlantID Backend on 0.0.0.0:%s", port)
    # DO NOT set debug=True in production
    app.run(host="0.0.0.0", port=port)



