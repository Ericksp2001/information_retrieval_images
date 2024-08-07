import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import tensorflow as tf

# Configuración del modelo base ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)

# Añadir capas superiores
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(102, activation='softmax')(x)  # Ajusta el número de clases a 102

# Crear el modelo combinado
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo con un optimizador y una función de pérdida
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

def extract_features(model, image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Configuración del directorio de datos
data_dir = os.path.join(os.path.dirname(__file__), '../data/101_ObjectCategories')

# Diccionarios para almacenar las características y categorías de cada imagen
train_index = {}
test_index = {}

# Itera a través de cada subdirectorio (cada subdirectorio representa una categoría) en el directorio de datos
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        image_paths = [os.path.join(class_path, image_name) for image_name in os.listdir(class_path)]
        train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
        
        for image_path in train_paths:
            features = extract_features(model, image_path)
            train_index[os.path.join(class_dir, os.path.basename(image_path))] = (features, class_dir)
        
        for image_path in test_paths:
            features = extract_features(model, image_path)
            test_index[os.path.join(class_dir, os.path.basename(image_path))] = (features, class_dir)

# Guarda los diccionarios de características en archivos utilizando pickle
with open('train_index.pkl', 'wb') as f:
    pickle.dump(train_index, f)

with open('test_index.pkl', 'wb') as f:
    pickle.dump(test_index, f)

print(f'Indexado {len(train_index)} imágenes para entrenamiento.')
print(f'Indexado {len(test_index)} imágenes para prueba.')
