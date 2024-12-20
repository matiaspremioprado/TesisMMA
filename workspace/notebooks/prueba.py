import gdown
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el enlace de Google Drive desde el archivo link_drive.txt
with open('workspace/resources/models/link_drive.txt', 'r') as f:
    model_link = f.read().strip()

# Descargar el archivo del modelo desde el enlace
model_path = 'model_downloaded.h5'
gdown.download(model_link, model_path, quiet=False)

# Cargar el modelo entrenado
model = load_model(model_path)

# Cargar las rutas de las imágenes del conjunto de prueba
test_dir = pathlib.Path('workspace/resources/datasets/output_folders/test')
test_folders = os.listdir(test_dir)

# Preparar el generador de imágenes para el conjunto de prueba
img_size = (224, 224)
batch_size = 64

# Crear un dataframe con las rutas de las imágenes de prueba
filepaths = []
labels = []

for folder in test_folders:
    folder_path = os.path.join(test_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):  # Filtrar por imagen
                filepaths.append(os.path.join(folder_path, file))
                labels.append(folder)

# Crear un dataframe de pandas para el conjunto de prueba
test_df = pd.DataFrame({
    'filepaths': filepaths,
    'labels': labels
})

# Cargar el generador de imágenes para el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepaths',
    y_col='labels',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluar el modelo
test_score = model.evaluate(test_gen, verbose=1)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# Performance del modelo (historia de entrenamiento)
# Esto solo se puede hacer si tienes los datos de entrenamiento previos, ya que history es un objeto del fit() previo
# Si no tienes la historia de entrenamiento, omite esta parte

# Aquí asumimos que tienes el historial de entrenamiento, pero si no, lo puedes omitir:
# Estos valores deberían haber sido guardados previamente, por ejemplo:
# tr_acc = history.history['accuracy']
# tr_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']

# Si no tienes los datos de entrenamiento, simplemente omite la parte de la gráfica
try:
    # Gráfico de performance del modelo
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Gráfico de la historia de entrenamiento
    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

except NameError:
    print("No se encontró el historial de entrenamiento, por lo que no se puede graficar.")
