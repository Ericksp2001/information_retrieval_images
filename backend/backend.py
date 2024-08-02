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

dictionary = {
    1: "accordion",
    2: "airplanes",
    3: "anchor",
    4: "ant",
    5: "background_google",
    6: "barrel",
    7: "bass",
    8: "beaver",
    9: "binocular",
    10: "bonsai",
    11: "brain",
    12: "brontosaurus",
    13: "buddha",
    14: "butterfly",
    15: "camera",
    16: "cannon",
    17: "car_side",
    18: "ceiling_fan",
    19: "cellphone",
    20: "chair",
    21: "chandelier",
    22: "cougar_body",
    23: "cougar_face",
    24: "crab",
    25: "crayfish",
    26: "crocodile",
    27: "crocodile_head",
    28: "cup",
    29: "dalmatian",
    30: "dollar_bill",
    31: "dolphin",
    32: "dragonfly",
    33: "electric_guitar",
    34: "elephant",
    35: "emu",
    36: "euphonium",
    37: "ewer",
    38: "faces",
    39: "faces_easy",
    40: "ferry",
    41: "flamingo",
    42: "flamingo_head",
    43: "garfield",
    44: "gerenuk",
    45: "gramophone",
    46: "grand_piano",
    47: "hawksbill",
    48: "headphone",
    49: "hedgehog",
    50: "helicopter",
    51: "ibis",
    52: "inline_skate",
    53: "joshua_tree",
    54: "kangaroo",
    55: "ketch",
    56: "lamp",
    57: "laptop",
    58: "leopards",
    59: "llama",
    60: "lobster",
    61: "lotus",
    62: "mandolin",
    63: "mayfly",
    64: "menorah",
    65: "metronome",
    66: "minaret",
    67: "motorbikes",
    68: "nautilus",
    69: "octopus",
    70: "okapi",
    71: "pagoda",
    72: "panda",
    73: "pigeon",
    74: "pizza",
    75: "platypus",
    76: "pyramid",
    77: "revolver",
    78: "rhino",
    79: "rooster",
    80: "saxophone",
    81: "schooner",
    82: "scissors",
    83: "scorpion",
    84: "sea_horse",
    85: "snoopy",
    86: "soccer_ball",
    87: "stapler",
    88: "starfish",
    89: "stegosaurus",
    90: "stop_sign",
    91: "strawberry",
    92: "sunflower",
    93: "tick",
    94: "trilobite",
    95: "umbrella",
    96: "watch",
    97: "water_lilly",
    98: "wheelchair",
    99: "wild_cat",
    100: "windsor_chair",
    101: "wrench",
    102: "yin_yang"
}

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