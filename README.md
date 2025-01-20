# Proyecto de Detección de Objetos con Computer Vision

## Descripción
Este proyecto tiene como objetivo entrenar un modelo de detección de objetos que pueda identificar logos de marcas en imágenes y videos. La aplicación principal es analizar la presencia de logos en videos publicitarios para medir el tiempo que aparecen en pantalla, proporcionando métricas clave para la toma de decisiones de marketing.

## Objetivos del Proyecto

### Nivel Esencial (Completado):
- Entrenar un modelo que detecte una nueva categoría (marca) y localice su posición dentro de un bounding box en imágenes.

### Nivel Medio (Completado):
- Extender la funcionalidad del modelo para procesar videos.
- Mostrar debajo de cada detección el nombre de la marca detectada.

### Niveles Futuros (Planeado):
- **Nivel Avanzado:**
  - Añadir el porcentaje de confianza en las detecciones.
  - Almacenar las detecciones en una base de datos con información relevante (nombre del video, bounding boxes, etc.).
  - Detectar múltiples marcas en el mismo video.
- **Nivel Experto:**
  - Crear una interfaz web para cargar videos y visualizar resultados.
  - Implementar un servicio en la nube con una API para acceder al modelo.

## Tecnologías Utilizadas
El proyecto emplea las siguientes tecnologías:

- **Modelos de Detección de Objetos:** YOLOv8n.
- **Frameworks de Deep Learning:** TensorFlow, PyTorch, TensorFlow/Keras.
- **Librerías Complementarias:** OpenCV, scikit-image, Pillow.

## Metodología de Trabajo
Se ha implementado la metodología de **Pair Programming**, donde:
- Un desarrollador actúa como el "Piloto" escribiendo el código.
- El otro actúa como "Copiloto", supervisando y aportando ideas en tiempo real.
- Los roles se alternan regularmente.

## Proceso de Desarrollo
1. **Obtención y Etiquetado de Datos:**
   - Recopilación de imágenes de logos de marcas.
   - Etiquetado manual utilizando herramientas como Roboflow.

2. **Entrenamiento del Modelo:**
   - Entrenamiento inicial para una marca específica con imágenes estáticas (nivel esencial).

3. **Extensión a Videos:**
   - Adaptación del modelo para analizar videos.
   - Adición de superposiciones que incluyan el nombre de la marca detectada.

4. **Pruebas y Evaluación:**
   - Evaluación del modelo en videos reales.
   - Ajustes en base a métricas de precisión y recall.

## Estructura del Proyecto
```
proyecto-cv-deteccion-objetos/
├── data/                # Datos de entrenamiento y validación
├── models/              # Modelos entrenados
├── scripts/             # Scripts para entrenamiento, inferencia y análisis
├── results/             # Resultados de las detecciones
├── docs/                # Documentación del proyecto
└── README.md            # Archivo README
```

## Instrucciones para Ejecutar el Proyecto
1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Entrena el modelo:
   ```bash
   python scripts/train.py --config configs/train_config.yaml
   ```
4. Realiza predicciones en imágenes:
   ```bash
   python scripts/predict.py --image_path data/test_image.jpg
   ```
5. Analiza videos:
   ```bash
   python scripts/video_analysis.py --video_path data/test_video.mp4
   ```

## Contribuciones
Las contribuciones al proyecto son bienvenidas. Sigue estos pasos:
1. Crea un fork del repositorio.
2. Crea una nueva rama para tu feature:
   ```bash
   git checkout -b feature/nueva_funcionalidad
   ```
3. Realiza tus cambios y haz un commit:
   ```bash
   git commit -m "Añade nueva funcionalidad"
   ```
4. Envía un pull request.

## Contacto
Para cualquier duda o sugerencia, puedes contactarnos en:
- Correo: soporte@proyectocv.com
- GitHub: [Enlace al repositorio](<URL_DEL_REPOSITORIO>)
