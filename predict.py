import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_image

model = load_model("model/traffic_sign_model.h5")

def predict_sign(image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return np.argmax(prediction), np.max(prediction)
