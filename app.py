import streamlit as st
import cv2
import numpy as np
from predict import predict_sign

st.set_page_config(page_title="Reconocimiento de Señales", layout="centered")

st.title("🚦 Reconocimiento de Señales de Tráfico")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    class_id, confidence = predict_sign(image)

    st.image(uploaded_file, width=250)
    st.success(f"Clase detectada: {class_id}")
    st.info(f"Confianza: {confidence:.2%}")
