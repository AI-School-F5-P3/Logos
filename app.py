import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch

# Configurar página
st.set_page_config(page_title="Detector de Logos Coca-Cola", layout="wide")

# Cargar el modelo
@st.cache_resource
def load_model():
    return YOLO('models/yolov8n.pt')

model = load_model()

# Función para procesar la imagen y realizar la detección
def detect_logo(image):
    # Realizar la predicción
    results = model(image)[0]
    
    # Convertir la imagen a numpy array si es necesario
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Dibujar las detecciones
    for box in results.boxes:
        bbox = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        
        # Dibujar el rectángulo y la confianza
        cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])), 
            (int(bbox[2]), int(bbox[3])), 
            (0, 255, 0), 
            2
        )
        cv2.putText(
            image,
            f'Coca-Cola: {conf:.2f}',
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return Image.fromarray(image), results

# Interfaz de usuario
st.title('Detección de Logos Coca-Cola')
st.write('Esta aplicación detecta logos de Coca-Cola en imágenes.')

# Upload de imagen
uploaded_file = st.file_uploader("Sube una imagen", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Cargar y mostrar la imagen original
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', use_column_width=True)
    
    # Botón para realizar la detección
    if st.button('Detectar Logo'):
        st.write('Procesando...')
        
        # Realizar la detección
        processed_image, results = detect_logo(image)
        
        # Mostrar resultados
        st.image(processed_image, caption='Detecciones', use_column_width=True)
        
        # Mostrar información de las detecciones
        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            st.write(f'Detección {i+1}: Confianza = {conf:.2f}')