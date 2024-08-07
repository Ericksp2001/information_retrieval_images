import os
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
import tensorflow as tf

# Función para cargar el índice
def load_index(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Función para preprocesar una imagen
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Cargar el índice de prueba
test_index_path = 'test_index.pkl'
test_index = load_index(test_index_path)

# Extraer las rutas de las imágenes y las etiquetas del índice de prueba
test_image_paths = [os.path.join(r'C:\Users\erick\Escritorio\information_retrieval_images\backend\data\101_ObjectCategories', k) for k in test_index.keys()]
test_labels = np.array([label for feat, label in test_index.values()])

# Preprocesar las imágenes de prueba
test_features = np.vstack([preprocess_image(p) for p in test_image_paths])

# Codificar etiquetas
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)
test_labels_categorical = to_categorical(test_labels_encoded)

# Cargar el modelo entrenado
model_path = r'C:\Users\erick\Escritorio\information_retrieval_images\backend\recursos\modelo_metrics\modelo_multinucleo.h5'
model = load_model(model_path)

# Evaluar el modelo
loss, accuracy = model.evaluate(test_features, test_labels_categorical)

# Hacer predicciones
predictions = model.predict(test_features)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)

# Calcular métricas
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')

# Imprimir reporte de clasificación
print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

# Guardar las métricas en un archivo de texto
def save_metrics_to_file(model_name, loss, accuracy, precision, recall, f1):
    with open(f'{model_name}_metrics.txt', 'w') as f:
        f.write(f'Modelo: {model_name}\n')
        f.write(f'Pérdida: {loss}\n')
        f.write(f'Precisión: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1-score: {f1}\n')

save_metrics_to_file('modelo_multinucleo', loss, accuracy, precision, recall, f1)
