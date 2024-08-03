import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Configuración del modelo
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Configuración del directorio de datos
data_dir = os.path.join(os.path.dirname(__file__), './data/101_ObjectCategories')  

# Diccionario para almacenar las características y categorías de cada imagen
index = {}

# Itera a través de cada subdirectorio (cada subdirectorio representa una categoría) en el directorio de datos
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            features = extract_features(base_model, image_path)
            index[os.path.join(class_dir, image_name)] = (features, class_dir)

# Guarda el diccionario de características en un archivo utilizando pickle
with open('index.pkl', 'wb') as f:
    pickle.dump(index, f)

print(f'Indexado {len(index)} imágenes.')