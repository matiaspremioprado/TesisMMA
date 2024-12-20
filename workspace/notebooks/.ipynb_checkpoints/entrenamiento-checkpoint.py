pip install tensorflow==2.9.1 opencv-python numpy pandas seaborn matplotlib

# System Libs
import gdown
import os
import time
import shutil
import pathlib
import itertools
from PIL import Image
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Leer los enlaces de Google Drive desde los archivos .txt
with open('workspace/resources/datasets/dataset_medicamentos/link_drive.txt', 'r') as f:
    google_drive_link = f.read().strip()

with open('workspace/resources/models/link_drive.txt', 'r') as f:
    model_save_path = f.read().strip()

# Descargar el archivo ZIP desde el link de Google Drive
gdown.download(google_drive_link, 'dataset_medicamentos.zip', quiet=False)

# Extraer el archivo ZIP
zip_path = 'dataset_medicamentos.zip'
data_dir = 'workspace/resources/datasets/dataset_medicamentos/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Verificación de que los archivos fueron extraídos
print(f'Contenido extraído: {os.listdir(data_dir)}')

# Ruta de las imágenes extraídas
data_dir = pathlib.Path(data_dir)

# Crear listas para las rutas de los archivos e imágenes
filepaths = []
labels = []

# Crear el DataFrame con las rutas y etiquetas
folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    filelist = os.listdir(foldpath)
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        filepaths.append(fpath)
        labels.append(fold)

# Concatenar las rutas de las imágenes y las etiquetas en un DataFrame
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)

# División en conjuntos de entrenamiento, validación y prueba
train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123)
valid_df, test_df = train_test_split(dummy_df,  train_size= 0.6, shuffle= True, random_state= 123)

# Verificación de la división de los datos
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")
print(f"Test set size: {len(test_df)}")

# Tamaño de la imagen y parámetros del modelo
batch_size = 64
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_df['labels'].unique()))

# Data Augmentation (función de ejemplo)
def scalar(img):
    return img

# Cargar generadores de imágenes
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, x_col='filepaths', y_col='labels', 
                                              target_size=img_size, batch_size=batch_size, class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = valid_datagen.flow_from_dataframe(dataframe=valid_df, x_col='filepaths', y_col='labels', 
                                              target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Modelado del modelo EfficientNetB1
base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet', input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(class_count, activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Entrenamiento del modelo
epochs = 3
history = model.fit(x=train_gen, epochs=epochs, validation_data=valid_gen, verbose=1)

# Guardar el modelo en Google Drive
model.save(model_save_path + '/Drug_Name.h5')
print(f'Modelo guardado en: {model_save_path}')

