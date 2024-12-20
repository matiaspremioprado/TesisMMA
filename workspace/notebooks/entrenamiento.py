import gdown
import zipfile
import os
import pathlib
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras import regularizers

# Leer el enlace de Google Drive desde el archivo link_drive.txt
with open('workspace/resources/datasets/dataset_medicamentos/link_drive.txt', 'r') as f:
    google_drive_link = f.read().strip()

with open('workspace/resources/models/link_drive.txt', 'r') as f:
    model_save_path = f.read().strip()

# Descargar el archivo ZIP desde el link de Google Drive
zip_path = 'dataset_medicamentos.zip'
gdown.download(google_drive_link, zip_path, quiet=False)

# Definir la ruta para extraer el contenido
data_dir = 'workspace/resources/datasets/dataset_medicamentos/'

# Extraer el archivo ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Verificación de que los archivos fueron extraídos
print(f'Contenido extraído: {os.listdir(data_dir)}')

# Convertir la ruta de data_dir a un objeto pathlib.Path
data_dir = pathlib.Path(data_dir)

# Crear listas para las rutas de los archivos e imágenes
filepaths = []
labels = []

# Filtrar para excluir archivos no válidos como .gitkeep
folds = os.listdir(data_dir)
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    if os.path.isdir(foldpath):  # Verificar si es un directorio
        filelist = os.listdir(foldpath)
        for file in filelist:
            # Excluir .gitkeep u otros archivos no deseados
            if not file.startswith('.') and os.path.isfile(os.path.join(foldpath, file)):
                fpath = os.path.join(foldpath, file)
                filepaths.append(fpath)
                labels.append(fold)

# Guardar las carpetas en workspace/resources/datasets
output_dataset_path = 'workspace/resources/datasets/output_folders/'
os.makedirs(output_dataset_path, exist_ok=True)

# Crear las carpetas base para train, valid y test
train_path = os.path.join(output_dataset_path, 'train')
valid_path = os.path.join(output_dataset_path, 'valid')
test_path = os.path.join(output_dataset_path, 'test')

# Crear las carpetas si no existen
os.makedirs(train_path, exist_ok=True)
os.makedirs(valid_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Función para mover las imágenes a las carpetas correspondientes
def save_images_to_folders(df, output_path):
    for i, filepath in enumerate(df['filepaths']):
        label = df['labels'].iloc[i]
        label_folder = os.path.join(output_path, label)
        os.makedirs(label_folder, exist_ok=True)  # Crear carpeta por clase si no existe
        shutil.copy(filepath, os.path.join(label_folder, os.path.basename(filepath)))

# Concatenar las rutas de las imágenes y las etiquetas en un DataFrame
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis= 1)

# División en conjuntos de entrenamiento, validación y prueba
train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.6, shuffle=True, random_state=123)

# Verificación de la división de los datos
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")
print(f"Test set size: {len(test_df)}")

# Guardar imágenes de entrenamiento, validación y prueba
save_images_to_folders(train_df, train_path)
save_images_to_folders(valid_df, valid_path)
save_images_to_folders(test_df, test_path)

print(f'Imágenes guardadas en las carpetas correspondientes dentro de {output_dataset_path}')

# Modelado del modelo EfficientNetB1
img_size = (224, 224)
img_shape = (img_size[0], img_size[1], 3)
batch_size = 64
class_count = len(list(train_df['labels'].unique()))

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
