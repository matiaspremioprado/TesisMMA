import os
import tensorflow as tf
import requests
from PIL import Image
import numpy as np

# Ruta donde se almacenará el modelo descargado si no está presente localmente
MODEL_PATH = 'workspace/resources/models/Drug_Name.h5'

# Función para leer la ruta del modelo desde link_drive.txt
def get_model_path():
    model_path_file = 'workspace/resources/models/link_drive.txt'
    with open(model_path_file, 'r') as f:
        return f.read().strip()

# Función para descargar el modelo desde Google Drive si no está en el servidor
def download_model_from_drive(drive_path: str, destination: str):
    # Implementar la lógica para descargar el modelo desde Google Drive usando la URL proporcionada
    response = requests.get(drive_path)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Modelo descargado en {destination}")
    else:
        print("Error al descargar el modelo.")

# Función para cargar el modelo
def load_model(model_path: str):
    if not os.path.exists(model_path):
        # Si el modelo no existe, intenta descargarlo
        model_link = get_model_path()
        download_model_from_drive(model_link, model_path)
    
    # Cargar el modelo desde el disco
    model = tf.keras.models.load_model(model_path)
    return model

# Cargar el modelo
model = load_model(MODEL_PATH)

# Función para validar la imagen (en cuanto a formato, tamaño, etc.)
def validate_image(image: bytes):
    try:
        img = Image.open(io.BytesIO(image))
        img.verify()  # Verifica si la imagen es válida
        img = img.convert('RGB')
        
        # Verifica que la imagen no sea demasiado grande
        if img.size[0] > 5000 or img.size[1] > 5000:
            return False, "La imagen es demasiado grande."

        return True, None
    except Exception as e:
        return False, f"Error al procesar la imagen: {str(e)}"

# Función de preprocesado de la imagen antes de la predicción
def preprocess_image(image: bytes):
    img = Image.open(io.BytesIO(image))
    img = img.convert('RGB')  # Convertir a RGB si es necesario
    img = img.resize((224, 224))  # Ajuste de tamaño para el modelo
    img_array = np.array(img)  # Convertir imagen a array
    img_array = np.expand_dims(img_array, axis=0)  # Expande dimensiones para el batch
    img_array /= 255.0  # Normalización (ajustar si es necesario)
    return img_array

# Realizar una predicción sobre una imagen cargada
def predict(image_bytes):
    # Validar la imagen
    valid, error_message = validate_image(image_bytes)
    if not valid:
        return error_message
    
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image_bytes)

    # Realizar la predicción
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    
    return predicted_class

# Test de predicción (puedes reemplazar el contenido con una imagen de prueba)
image_path = "path_to_your_image.jpg"
with open(image_path, "rb") as img_file:
    image_bytes = img_file.read()

# Realizar la predicción
predicted_class = predict(image_bytes)
print(f"Predicción de la clase: {predicted_class}")
