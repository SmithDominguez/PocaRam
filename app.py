import os
import cv2
import numpy as np
import face_recognition
import gc
from flask import Flask, render_template, request
import urllib.request
from PIL import Image

# Asegúrate de que la carpeta 'static/uploads' exista
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Configuración
URLS_CELEBRIDADES = {
    "Messi": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg/250px-Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg",
    "Shakira": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/2023-11-16_Gala_de_los_Latin_Grammy%2C_03_%28cropped%2901.jpg/330px-2023-11-16_Gala_de_los_Latin_Grammy%2C_03_%28cropped%2901.jpg"
}
TOLERANCIA = 0.6
HEADERS = {'User-Agent': 'Mozilla/5.0'}

app = Flask(__name__)

# Configuración para subir archivos
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Redimensionar la imagen para reducir su tamaño en memoria
def redimensionar_imagen(path, new_width=800, new_height=800):
    with Image.open(path) as img:
        img = img.resize((new_width, new_height))  # Cambiar tamaño a 800x800
        img.save(path, "JPEG", quality=75)  # Comprimir imagen

def cargar_celebridades():
    nombres = []
    embeddings = []
    for nombre, url in URLS_CELEBRIDADES.items():
        try:
            data = descargar_imagen(url)
            img_array = np.asarray(bytearray(data), dtype=np.uint8)
            imagen = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            caras = face_recognition.face_encodings(rgb)
            if len(caras) == 0:
                continue
            nombres.append(nombre)
            embeddings.append(caras[0])
        except Exception as e:
            continue
    return nombres, embeddings

def descargar_imagen(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as response:
        data = response.read()
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detectar', methods=['POST'])
def detectar():
    # Verificar si el usuario subió un archivo
    if 'file' not in request.files:
        return "No se ha seleccionado un archivo", 400
    file = request.files['file']
    if file.filename == '':
        return "No se seleccionó un archivo", 400
    if file and allowed_file(file.filename):
        # Guardar el archivo subido en el servidor
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Redimensionar la imagen para ahorrar memoria
        redimensionar_imagen(filename)

        # Cargar celebridades
        nombres_conocidos, embeddings_conocidos = cargar_celebridades()
        if not nombres_conocidos:
            return "No se cargaron celebridades", 400

        # Procesar la imagen subida
        imagen_prueba = cv2.imread(filename)
        rgb_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2RGB)

        ubicaciones_caras = face_recognition.face_locations(rgb_prueba)
        embeddings_prueba = face_recognition.face_encodings(rgb_prueba, ubicaciones_caras)

        for (top, right, bottom, left), embedding in zip(ubicaciones_caras, embeddings_prueba):
            coincidencias = face_recognition.compare_faces(embeddings_conocidos, embedding, tolerance=TOLERANCIA)
            distancias = face_recognition.face_distance(embeddings_conocidos, embedding)
            nombre_detectado = "Desconocido"
            if True in coincidencias:
                indice = np.argmin(distancias)
                nombre_detectado = nombres_conocidos[indice]

            cv2.rectangle(imagen_prueba, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(imagen_prueba, nombre_detectado, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Guardar la imagen procesada
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, imagen_prueba)

        # Eliminar el archivo subido después de procesarlo
        os.remove(filename)  # Elimina el archivo original subido

        # Liberar memoria
        del imagen_prueba
        gc.collect()

        return render_template('result.html', image_path=result_path)

    return "Archivo no permitido", 400

if __name__ == "__main__":
    app.run(debug=True)
