from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from PIL import Image
import io
import tensorflow as tf
import requests

app = FastAPI()

# Ruta para obtener la ruta del modelo desde link_drive.txt
MODEL_PATH = 'workspace/resources/models/Drug_Name.h5'

# Función para leer la ruta del modelo desde link_drive.txt
def get_model_path():
    model_path = 'workspace/resources/models/link_drive.txt'
    with open(model_path, 'r') as f:
        return f.read().strip()

# Función para descargar el modelo desde Google Drive si no está en el servidor
def download_model_from_drive(drive_path: str, destination: str):
    # Esto debería implementar la descarga desde Google Drive, por ejemplo, usando el ID del archivo.
    # Aquí se puede usar gdown o cualquier otro método para descargar desde Google Drive
    # Suponiendo que el archivo es accesible por URL directa.
    
    # Check
    response = requests.get(drive_path)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Modelo descargado en {destination}")
    else:
        print("Error al descargar el modelo.")

# Función para cargar el modelo desde disco
def load_model(model_path: str):
    if not os.path.exists(model_path):
        # Si el modelo no existe, intenta descargarlo
        model_link = get_model_path()
        download_model_from_drive(model_link, model_path)
    
    # Cargar el modelo guardado
    model = tf.keras.models.load_model(model_path)
    return model

# Cargar el modelo al inicio de la API
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

class PredictionRequest(BaseModel):
    image: UploadFile

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Obtener la imagen del request
    image = await request.image.read()
    
    # Validar la imagen
    valid, error_message = validate_image(image)
    if not valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image)

    # Realizar la predicción
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    
    # Devolver la clase predicha (puedes modificar esto según el formato que necesites)
    return {"predicted_class": int(predicted_class)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
