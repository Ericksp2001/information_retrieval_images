import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from multiprocessing import Pool, cpu_count

# Cargar el modelo
def load_model_shared():
    global base_model
    base_model = load_model(r'C:\Users\erick\Escritorio\information_retrieval_images\backend\recursos\modelo_metrics\modelo_multinucleo.h5')

# Extraer características de una imagen
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    features = base_model.predict(img_array)
    return features.flatten()

# Procesar una imagen
def process_image(args):
    class_dir, image_name, data_dir = args
    image_path = os.path.join(data_dir, class_dir, image_name)
    features = extract_features(image_path)
    return (os.path.join(class_dir, image_name), (features, class_dir))

if __name__ == '__main__':
    # Configuración del directorio de datos
    data_dir = os.path.join(os.path.dirname(__file__), '../data/101_ObjectCategories')

    # Lista para almacenar los argumentos para el Pool
    args_list = []

    # Iterar a través de cada subdirectorio (cada subdirectorio representa una categoría) en el directorio de datos
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                args_list.append((class_dir, image_name, data_dir))

    # Inicializar el Pool de procesos y cargar el modelo en cada proceso
    with Pool(processes=cpu_count(), initializer=load_model_shared) as pool:
        results = pool.map(process_image, args_list)

    # Diccionario para almacenar las características y categorías de cada imagen
    index = dict(results)

    # Guardar el diccionario de características en un archivo utilizando pickle
    with open('index.pkl', 'wb') as f:
        pickle.dump(index, f)

    print(f'Indexado {len(index)} imágenes.')