import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model # type: ignore

# Configuración del modelo
base_model = load_model(r'C:\Users\erick\Escritorio\information_retrieval_images\backend\recursos\modelo_metrics\modelo_multinucleo.h5')

def extract_features(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def search_similar(image_path, index, category=None, top_k=30):
    if image_path:
        query_features = extract_features(base_model, image_path)
        similarities = {
            k: cosine_similarity([query_features], [v[0]]).flatten()[0]
            for k, v in index.items()
            if category is None or v[1] == category
        }
    else:
        similarities = {k: 1 for k, v in index.items() if category is None or v[1] == category}
    
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:top_k]