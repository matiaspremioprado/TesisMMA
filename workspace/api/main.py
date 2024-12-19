from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from utils import validate_image, preprocess_image, download_model, load_model, get_model_link

app = FastAPI()

# Ruta para obtener el enlace de descarga del modelo
MODEL_PATH = 'workspace/resources/models/Drug_Name.h5'

# Cargar el modelo al inicio de la API
model = load_model(MODEL_PATH)

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
    
    # Devolver la clase predicha
    return {"predicted_class": int(predicted_class)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Ruta para obtener el estado de la API
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}
