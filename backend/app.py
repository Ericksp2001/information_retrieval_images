from flask import Flask, request, jsonify, send_from_directory, abort, url_for
import os
import pickle
import tempfile
from image_search import search_similar
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configura el directorio donde se guardan las imágenes
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(__file__), './data/101_ObjectCategories')  

# Cargar el índice desde el archivo
index_path = os.path.join(os.path.dirname(__file__), './recursos/Index_Generation/index.pkl')
index = None
if os.path.exists(index_path):
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    print(f'Índice cargado desde {index_path}.')
    print(f'Número de imágenes indexadas: {len(index)}')
else:
    print(f'No se encontró el archivo {index_path}.')

# Ruta para servir imágenes desde el directorio 'static/caltech-101'
@app.route('/data/101_ObjectCategories/<category>/<filename>')
def image_file(category, filename):
    file_path = os.path.join(app.config['IMAGE_FOLDER'], category, filename)
    if os.path.exists(file_path):
        return send_from_directory(os.path.join(app.config['IMAGE_FOLDER'], category), filename)
    else:
        abort(404)  # Devolver error 404 si el archivo no se encuentra

# Ruta para buscar imágenes similares a una imagen cargada
@app.route('/search_image', methods=['POST'])
def search_image():
    if index is None:
        return jsonify(error="El índice no se ha cargado correctamente."), 500

    file = request.files.get('file')
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file_path = temp_file.name
        try:
            # Guarda el archivo cargado en el directorio temporal
            file.save(file_path)
            print(f"Archivo guardado temporalmente en {file_path}")

            if not os.path.isfile(file_path):
                return jsonify(error="No se pudo guardar el archivo."), 500

            # Busca imágenes similares
            similar_images = search_similar(file_path, index)
            print(f"Imágenes similares encontradas: {similar_images}")

            # Procesa las imágenes similares
            processed_images = []
            for img_path, sim in similar_images:
                img_name = os.path.basename(img_path)
                if img_path in index:
                    category = index[img_path][1]
                    processed_images.append({
                        "path": url_for('image_file', category=category, filename=img_name),
                        "similarity": float(sim)  # Convertir float32 a float
                    })

            print(f"Imágenes similares procesadas: {processed_images}")

            return jsonify(query_image=temp_file.name, similar_images=processed_images)
        except Exception as e:
            print(f"Error al procesar la imagen: {str(e)}")
            return jsonify(error=str(e)), 500
        finally:
            os.unlink(file_path)  # Mover la eliminación del archivo al bloque finally para asegurar que se cierre correctamente

    return jsonify(error="No se recibió ningún archivo."), 400

if __name__ == '__main__':
    app.run(debug=True)
