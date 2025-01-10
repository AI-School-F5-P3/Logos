import os
import requests
from tqdm import tqdm

# Rutas de los archivos descargados manualmente
annotations_file = 'FlickrLogos-27-dataset-master/flickr_logos_27_dataset_training_set_annotation.txt'
urls_file = 'FlickrLogos-27-dataset-master/flickr_logos_27_dataset_distractor_set_urls.txt'
output_dir = 'organized_data'
distractor_dir = 'distractor_images'

# Crear directorios de salida si no existen
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(distractor_dir):
    os.makedirs(distractor_dir)

# Función para descargar una imagen
def download_image(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

# Función para descargar imágenes distractoras
def download_distractor_images(urls_file, distractor_dir):
    if not os.path.exists(urls_file):
        print(f"Error: El archivo de URLs {urls_file} no existe.")
        return

    with open(urls_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            url = line.strip()
            image_name = f"distractor_{i}.jpg"
            image_path = os.path.join(distractor_dir, image_name)
            if not os.path.exists(image_path):
                download_image(url, image_path)

# Función para organizar las imágenes en carpetas por clase
def organize_images(annotations_file, output_dir):
    if not os.path.exists(annotations_file):
        print(f"Error: El archivo de anotaciones {annotations_file} no existe.")
        return

    with open(annotations_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Error: Línea inválida en el archivo de anotaciones: {line.strip()}")
                continue
            image_name = parts[0]
            class_name = parts[1]
            class_dir = os.path.join(output_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            image_path = os.path.join(class_dir, image_name)
            # Asumimos que las imágenes ya están descargadas en el directorio de distractores
            source_path = os.path.join(distractor_dir, image_name)
            if os.path.exists(source_path):
                os.rename(source_path, image_path)

# Descargar imágenes distractoras
download_distractor_images(urls_file, distractor_dir)

# Organizar las imágenes según las anotaciones
organize_images(annotations_file, output_dir)

print("Datos preparados y organizados en carpetas por clase.")