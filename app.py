from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)


# Config
UPLOAD_FOLDER = "static/upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "BrainTumor_multiclass.h5"
IMG_SIZE = 64


classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# Load Model
model = load_model(MODEL_PATH)


# Image Preprocessing
def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    results = []

    if "file[]" not in request.files:
        return "No files uploaded"

    files = request.files.getlist("file[]")

    for file in files:
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        preds = model.predict(img, verbose=0)[0]

        class_index = np.argmax(preds)
        predicted_class = classes[class_index]
        confidence = round(float(preds[class_index]) * 100, 2)

        # Full class probabilities
        all_probs = {
            classes[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(classes))
        }

        results.append({
            "filename": filename,
            "prediction": predicted_class,
            "confidence": confidence,
            "all_probs": all_probs
        })

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)