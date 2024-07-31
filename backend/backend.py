from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import numpy as np
from joblib import load
import tensorflow as tf
import os

# Directorio actual ""

app = Flask(__name__)
CORS(app)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Paths
train_features_path = os.path.join(os.path.dirname(__file__), '../recursos/train_features.npy')
train_labels_path = os.path.join(os.path.dirname(__file__), '../recursos/train_labels.npy')
nn_model_path = os.path.join(os.path.dirname(__file__), '../recursos/nearest_neighbors_model.joblib')

photos_path = os.path.join(os.path.dirname(__file__), '../data/downloads/extracted/TAR_GZ.101_ObjectCategories.tar.gz/101_ObjectCategories/')

# Load the NearestNeighbors model
nn_model = load(nn_model_path)


# Load the training features and labels
train_features_flat = np.load(train_features_path)
train_labels_flat = np.load(train_labels_path)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(img_array):
    features = model.predict(img_array)
    return features.flatten().reshape(1, -1)

@app.route('/search', methods=['POST'])
def search_similar_images():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    if file:
        # Save the file temporarily
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        # Preprocess the image and extract features
        img_array = preprocess_image(temp_path)
        query_features = extract_features(img_array)
        
        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(query_features)
        
        # Get the labels of the nearest neighbors
        nearest_labels = train_labels_flat[indices[0]]
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        # Prepare the response
        results = [{'index': int(idx), 'label': int(label), 'distance': float(dist)} for idx, label, dist in zip(indices[0], nearest_labels, distances[0])]
        
        return jsonify({'results': results})

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(photos_path, filename)

if __name__ == '__main__':
    app.run(debug=True)