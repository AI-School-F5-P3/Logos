import streamlit as st
from ultralytics import YOLO
import cv2
import sqlite3
import time

# Configurar página
st.set_page_config(page_title="Detector de Logos Coca-Cola en Video", layout="wide")

# Establecer umbral de confianza
CONFIDENCE_THRESHOLD = 0.6  # Ajusta este valor según tus necesidades

# Conectar a la base de datos SQLite
conn = sqlite3.connect('detections.db')
cursor = conn.cursor()

# Crear tabla si no existe
cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    logo_count INTEGER,
                    duration REAL)''')

# Cargar el modelo
@st.cache_resource
def load_model():
    return YOLO('models/best_v2.pt')

model = load_model()

# Función para procesar el video y realizar la detección
def detect_logo_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("No se pudo abrir el video.")
        return

    stframe = st.empty()  # Contenedor para mostrar los fotogramas procesados

    total_logo_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar la detección en cada fotograma
        results = model(frame)[0]
        frame_logo_count = 0

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= CONFIDENCE_THRESHOLD:
                frame_logo_count += 1
                bbox = box.xyxy[0].cpu().numpy()
                # Dibujar el rectángulo y la confianza
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f'Coca-Cola: {conf:.2f}',
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        total_logo_count += frame_logo_count

        # Convertir el fotograma de BGR a RGB para mostrar en Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)

    cap.release()

    # Calcular el tiempo total transcurrido
    total_duration = time.time() - start_time

    # Guardar los resultados en la base de datos
    cursor.execute('INSERT INTO detections (timestamp, logo_count, duration) VALUES (?, ?, ?)',
                   (time.strftime('%Y-%m-%d %H:%M:%S'), total_logo_count, total_duration))
    conn.commit()

    # Mostrar el total de detecciones y el tiempo total
    st.write(f"Detección completada. Se encontraron un total de {total_logo_count} logos en un tiempo total de {total_duration:.2f} segundos.")

# Interfaz de usuario
st.title('Detección de Logos Coca-Cola en Video')
st.write('Esta aplicación detecta logos de Coca-Cola en videos.')

# Upload de video
uploaded_video = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None:
    with open('uploaded_video.mp4', 'wb') as f:
        f.write(uploaded_video.read())

    # Botón para realizar la detección
    if st.button('Detectar Logo en Video'):
        st.write('Procesando video...')
        detect_logo_in_video('uploaded_video.mp4')
