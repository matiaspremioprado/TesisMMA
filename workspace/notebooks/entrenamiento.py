import gdown
import zipfile
import os

# Leer el enlace de Google Drive desde el archivo link_drive.txt
with open('workspace/resources/datasets/dataset_medicamentos/link_drive.txt', 'r') as f:
    google_drive_link = f.read().strip()

# Descargar el archivo ZIP desde el enlace de Google Drive
zip_path = 'dataset_medicamentos.zip'
gdown.download(google_drive_link, zip_path, quiet=False)

# Verificar si el archivo descargado es un ZIP válido
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.testzip()  # Verifica que el archivo ZIP no esté dañado
        print("El archivo ZIP es válido.")
        zip_ref.extractall('workspace/resources/datasets/dataset_medicamentos/')
        print(f'Contenido extraído: {os.listdir("workspace/resources/datasets/dataset_medicamentos/")}')
except zipfile.BadZipFile:
    print(f'Error: El archivo descargado no es un archivo ZIP válido. Verifica el enlace de Google Drive.')
except Exception as e:
    print(f'Ocurrió un error al intentar abrir el archivo ZIP: {e}')


