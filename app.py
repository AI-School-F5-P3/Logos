import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Cargar el modelo
model = tf.keras.models.load_model('logo_detector.h5')

# Función para preprocesar la imagen
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Función para realizar la detección
def detect_logo(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return class_idx, confidence

# Interfaz de usuario con Streamlit
st.title('Detección de Logos')
uploaded_file = st.file_uploader('Sube una imagen', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagen subida', use_column_width=True)
    st.write('')
    st.write('Detectando...')

    class_idx, confidence = detect_logo(img)
    st.write(f'Clase detectada: {class_idx}')
    st.write(f'Confianza: {confidence:.2f}')