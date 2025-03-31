import os
import zipfile
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import time
import shutil

# Definir rutas
base_path = "workspace/resources"
credentials_path = os.path.join(base_path, "credentials", "google_drive_credentials.json")  # Ruta al archivo credentials.json
config_path = os.path.join(base_path, "credentials", "google_drive_id_dataset_medicamentos.txt")
zip_path = os.path.join(base_path, "datasets", "dataset_medicamentos.zip")
extract_path = os.path.join(base_path, "datasets", "dataset_medicamentos")

# Leer el ID del archivo desde el archivo de configuración
with open(config_path, "r") as file:
    file_id = file.read().strip()

# Autenticación con Google Drive API
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def authenticate_google_drive():
    creds = None
    token_path = os.path.join(base_path, "credentials", "google_drive_token.json")  # Ruta al archivo token.json
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return creds

# Descargar el archivo usando la API de Google Drive
def download_file_from_google_drive(file_id, destination):
    creds = authenticate_google_drive()
    service = build("drive", "v3", credentials=creds)
    
    # Solicitar el archivo usando la API
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Descargando... {int(status.progress() * 100)}%")
    
    print(f"Archivo descargado en: {destination}")

def safe_delete_file(file_path, max_attempts=3):
    """Intenta eliminar un archivo con múltiples intentos y manejo de errores"""
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            print(f"✓ Archivo eliminado: {file_path}")
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(1)  # Espera 1 segundo entre intentos
                continue
            print(f"⚠️ No se pudo eliminar (intentos agotados): {file_path}")
            return False
        except Exception as e:
            print(f"⚠️ Error inesperado al eliminar: {e}")
            return False

# Descargar el archivo ZIP
download_file_from_google_drive(file_id, zip_path)

# Verificar si el archivo es un ZIP válido antes de intentar extraerlo
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Verificar si el archivo es un ZIP válido
        if zip_ref.testzip() is None:
            print(f"Archivos extraídos en: {extract_path}")
            zip_ref.extractall(extract_path)  # Extraer archivos
        else:
            print("Error: El archivo ZIP contiene archivos corruptos.")
except zipfile.BadZipFile:
    print("Error: El archivo descargado no es un ZIP válido o está corrupto.")
    # Verificar si el archivo es un HTML (página de advertencia de Google Drive)
    with open(zip_path, "r", encoding="utf-8") as f:
        content = f.read(500)  # Leer los primeros 500 caracteres
        if "<html>" in content.lower():
            print("Parece que Google Drive devolvió una página HTML en lugar del archivo.")
            print("Posible solución: Usa la API de Google Drive o verifica los permisos del archivo.")

# Eliminar el ZIP después de extraer (con manejo robusto)
if os.path.exists(zip_path):
    if not safe_delete_file(zip_path):
        print("\n🔍 Intentando métodos alternativos...")
        try:
            # Método alternativo 1: shutil
            os.unlink(zip_path)  # Equivalente a os.remove
            print("✓ Eliminado con os.unlink()")
        except:
            try:
                # Método alternativo 2: renombrar y eliminar después
                temp_name = zip_path + ".tmp"
                os.rename(zip_path, temp_name)
                os.remove(temp_name)
                print("✓ Eliminado después de renombrar")
            except Exception as e:
                print(f"✖ No se pudo eliminar el archivo. Error final: {e}")
                print(f"Por favor elimina manualmente: {os.path.abspath(zip_path)}")
else:
    print("ℹ️ El archivo ZIP ya no existe")