import os
from ultralytics import YOLO
import torch
import yaml
import time

# Configurar rutas - Actualizadas para usar rutas relativas
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

# Asegurar que el directorio models existe
os.makedirs(MODEL_PATH, exist_ok=True)

# Funciones auxiliares
def list_directory_contents(path):
    """Lista el contenido de un directorio y sus subdirectorios"""
    try:
        print(f"\nContenido de {path}:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")
    except Exception as e:
        print(f"Error al listar directorio {path}: {str(e)}")

def verify_dataset():
    """Verifica que el dataset esté correctamente estructurado"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    splits = ['train', 'valid', 'test']
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No se encuentra el directorio de datos en {DATASET_PATH}")
        return False

    for split in splits:
        img_path = os.path.join(DATASET_PATH, f'{split}/images')
        label_path = os.path.join(DATASET_PATH, f'{split}/labels')

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"❌ Error: Faltan directorios para {split}")
            return False

        images = [f for f in os.listdir(img_path) if f.lower().endswith(image_extensions)]
        labels = [f for f in os.listdir(label_path) if f.endswith('.txt')]

        print(f"\nContenido de {split}:")
        print(f"📸 Imágenes encontradas: {len(images)}")
        print(f"🏷️ Labels encontrados: {len(labels)}")

        if len(images) == 0:
            print(f"❌ Error: No hay imágenes en {split}")
            return False

    return True

def train_model(epochs=4):
    """Entrena el modelo YOLO"""
    print("\nVerificando dataset antes del entrenamiento...")
    if not verify_dataset():
        print("❌ Entrenamiento cancelado: Problemas con el dataset")
        return None

    print("\n🚀 Iniciando entrenamiento...")
    try:
        # Inicializar modelo
        model = YOLO('yolov8n.pt')

        # Configurar entrenamiento
        results = model.train(
            data=os.path.join(DATASET_PATH, 'data.yaml'),
            epochs=epochs,
            imgsz=640,
            batch=16,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=MODEL_PATH,
            name='logo_detection'
        )

        print("✅ Entrenamiento completado exitosamente")
        return results
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        return None

# Configurar/actualizar data.yaml
yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
if not os.path.exists(yaml_path):
    print(f"\nCreando nuevo archivo data.yaml en: {yaml_path}")
    data_yaml_content = {
        'path': DATASET_PATH,  # Actualizando path para usar ruta absoluta
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['pepsi']
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    print("Archivo data.yaml creado exitosamente")
else:
    print(f"\nArchivo data.yaml ya existe en: {yaml_path}")
    update_yaml = input("¿Deseas actualizar el archivo data.yaml con la clase 'pepsi'? (s/n): ")
    if update_yaml.lower() == 's':
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        yaml_content['names'] = ['pepsi']
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print("Archivo data.yaml actualizado con la clase 'pepsi'")

def test_model(image_path, conf_threshold=0.25):
    """Prueba el modelo entrenado"""
    model_path = os.path.join(MODEL_PATH, 'logo_detection/weights/best.pt')
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo en {model_path}")
        return None
        
    model = YOLO(model_path)
    results = model(image_path)[0]

    # Convertir resultados a formato más legible
    detections = []
    for box in results.boxes:
        detection = {
            'bbox': box.xyxy[0].cpu().numpy(),
            'confidence': float(box.conf[0]),
            'class_name': model.names[int(box.cls[0])]
        }
        if detection['confidence'] >= conf_threshold:
            detections.append(detection)

    return detections

if __name__ == "__main__":
    # Iniciar entrenamiento
    train_input = input("\n¿Deseas comenzar el entrenamiento ahora? (s/n): ")
    if train_input.lower() == 's':
        results = train_model()
        if results is not None:
            print("\n✅ Entrenamiento completado. El modelo está listo para usar.")
        else:
            print("\n❌ El entrenamiento no se completó debido a errores.")