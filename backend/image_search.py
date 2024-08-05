import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity

# Configuración del modelo
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

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