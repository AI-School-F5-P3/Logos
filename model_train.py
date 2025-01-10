import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Configuración
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
ANNOTATIONS_FILE = 'FlickrLogos-27-dataset-master/flickr_logos_27_dataset_training_set_annotation.txt'
URLS_FILE = 'FlickrLogos-27-dataset-master/flickr_logos_27_dataset_distractor_set_urls.txt'

# Leer las anotaciones y las URLs
annotations = pd.read_csv(ANNOTATIONS_FILE, sep='\t', header=None, names=['filename', 'class'])
urls = pd.read_csv(URLS_FILE, header=None, names=['url'])

# Crear un diccionario de URLs para las imágenes
url_dict = {url.split('/')[-1]: url for url in urls['url']}

# Generador de datos personalizado
class URLImageDataGenerator(Sequence):
    def __init__(self, annotations, url_dict, batch_size, img_size, num_classes):
        self.annotations = annotations
        self.url_dict = url_dict
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.classes = sorted(annotations['class'].unique())
        self.class_indices = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return int(np.ceil(len(self.annotations) / self.batch_size))

    def __getitem__(self, idx):
        batch_annotations = self.annotations[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for _, row in batch_annotations.iterrows():
            img_url = self.url_dict.get(row['filename'])
            if img_url:
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content)).resize((self.img_size, self.img_size))
                img_array = np.array(img) / 255.0
                batch_images.append(img_array)
                label = self.class_indices[row['class']]
                batch_labels.append(tf.keras.utils.to_categorical(label, num_classes=self.num_classes))
        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)

# Crear el generador de datos
num_classes = len(annotations['class'].unique())
train_gen = URLImageDataGenerator(annotations, url_dict, BATCH_SIZE, IMG_SIZE, num_classes)

# Modelo base
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_gen, epochs=EPOCHS)

# Guardar el modelo
model.save('logo_detector.h5')