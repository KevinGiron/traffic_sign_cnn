import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from model import build_model
from utils import preprocess_image
import cv2

DATA_DIR = "data/Train"
NUM_CLASSES = 43

X, y = [], []

for label in range(NUM_CLASSES):
    path = os.path.join(DATA_DIR, str(label))
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        img = preprocess_image(img)
        X.append(img)
        y.append(label)

X = np.array(X)
y = to_categorical(y, NUM_CLASSES)

model = build_model()
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

model.save("model/traffic_sign_model.h5")
