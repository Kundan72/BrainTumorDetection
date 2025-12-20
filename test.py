import cv2
import os
import numpy as np
from PIL import Image
# from keras.models import load_model
from tensorflow.keras.models import load_model

# Classes used during training
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
INPUT_SIZE = 64

MODEL_PATH = "BrainTumor_multiclass.h5"
TEST_FOLDER = r"Brain_tumor\Testing"

print("Loading model...")
model = load_model(MODEL_PATH)

# Ensure folder exists
if not os.path.exists(TEST_FOLDER):
    print("Testing folder path not found:", TEST_FOLDER)
    exit()

print(f"\nScanning testing folder: {TEST_FOLDER}\n")

# Loop through each class folder
for class_name in CLASS_NAMES:
    class_folder = os.path.join(TEST_FOLDER, class_name)

    if not os.path.exists(class_folder):
        print(f"Class folder missing: {class_folder}")
        continue

    print(f"\n=====Testing class: {class_name} =====")

    # List all images inside current class folder
    images = [f for f in os.listdir(class_folder)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(images) == 0:
        print("No images found in this folder.")
        continue

    for img_name in images:
        img_path = os.path.join(class_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Could not read image: {img_name}")
            continue

        # Preprocess
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)
        predicted_label = CLASS_NAMES[np.argmax(pred)]

        print(f"{img_name}  →  Predicted: {predicted_label}")

print("\nFinished predictions for all testing images!")