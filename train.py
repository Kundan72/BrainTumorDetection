import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Dataset paths
DATA_DIR = "Brain_tumor/"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = 64  # Small CNN


# Load dataset
def load_dataset(directory):
    dataset = []
    labels = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(directory, class_name)
        print("Loading:", class_path)  # DEBUG LINE ✅
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
                dataset.append(np.array(img))
                labels.append(idx)
    return np.array(dataset), np.array(labels)

X_train, y_train = load_dataset(TRAIN_DIR)
X_test, y_test = load_dataset(TEST_DIR)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# ------------------------
# Data augmentation
# ------------------------
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)


# Build CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=25,
    verbose=1
)


# Save model
model.save('BrainTumor_multiclass.h5')
print("Model saved as BrainTumor_multiclass.h5")