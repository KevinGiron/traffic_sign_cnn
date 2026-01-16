import cv2
import numpy as np

IMG_SIZE = 32

def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image
