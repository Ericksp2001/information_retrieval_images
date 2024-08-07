import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def train_model(model_name):
    # Usar MirroredStrategy para distribuir el entrenamiento entre múltiples núcleos de CPU
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
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

        # Definir generadores de datos para entrenamiento y validación
        train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(
            r'C:\Users\erick\Escritorio\information_retrieval_images\backend\data\101_ObjectCategories',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            r'C:\Users\erick\Escritorio\information_retrieval_images\backend\data\101_ObjectCategories',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        # Añadir un callback para guardar el mejor modelo
        checkpoint = ModelCheckpoint(f'{model_name}.h5', monitor='val_accuracy', save_best_only=True, mode='max')

        # Entrenar el modelo
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            epochs=10,
            callbacks=[checkpoint]
        )

        # Evaluar el modelo en el conjunto de validación
        loss, accuracy = model.evaluate(validation_generator)
        
        # Obtener predicciones
        val_generator = validation_generator
        val_generator.reset()
        predictions = model.predict(val_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator.classes

        # Imprimir reporte de clasificación
        print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    # Entrenar el modelo usando todos los núcleos de la CPU
    train_model('modelo_multinucleo')
