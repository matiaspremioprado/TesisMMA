import os
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Definir rutas
base_path = "workspace/resources"
credentials_path = os.path.join(base_path, "credentials", "credentials.json")  # Ruta al archivo credentials.json
config_path = os.path.join(base_path, "credentials", "google_drive_id_diccionario_medicamentos.txt")
csv_path = os.path.join(base_path, "datasets", "diccionario_medicamentos.csv")

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

# Descargar el archivo desde Google Drive
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

# Verificar y leer el archivo CSV
def verify_and_show_csv(csv_path):
    try:
        # Intentar abrir y leer el CSV
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        print(f"✅ Archivo CSV verificado correctamente con codificación ISO-8859-1.")
        
        # Mostrar las primeras filas del CSV
        print("Primeras filas del archivo CSV:")
        print(df.head())
    except UnicodeDecodeError:
        print(f"❌ Error al verificar el archivo CSV: No se puede leer con codificación ISO-8859-1.")
    except Exception as e:
        print(f"❌ Error al verificar el archivo CSV: {e}")

# Descargar el archivo CSV
download_file_from_google_drive(file_id, csv_path)

# Verificar el archivo CSV y mostrar las primeras filas
verify_and_show_csv(csv_path)
