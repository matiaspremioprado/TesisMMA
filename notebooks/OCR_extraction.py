import os
import random
import base64
import pandas as pd
from together import Together
from tqdm import tqdm

# Configuración global
DATASET_PATH = "workspace/resources/datasets/dataset_medicamentos"
OUTPUT_DIR = "workspace/resources/outputs"  # Nueva variable para el directorio de salida
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "resultados_extraccion_medicamentos.csv")  # Ruta completa
MODEL_NAME = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"  # Modelo seleccionado
SAMPLE_SIZE = 100  # Número de imágenes a procesar


# Prompt optimizado para medicamentos
PROMPT_INSTRUCCIONES = "Extract JUST the full medication title, don't add anything you don't see. Don't extract symbols like '*'. You should be just finding ONE of the following names: 'Pervinox Incoloro', 'Nopucid Shampoo Anti-Liendres', 'Neumotide', 'Nopucid Ultradim', 'Kanbis Cannabidiol 100 mg/ml', 'EVAPLAN digital', 'EVAPLAN Simple', 'Neumoterol 200', 'Nopucid IVER', 'Pervinox Jabon'"


def verificar_api_key():
    """Verifica y muestra el estado de la API Key"""
    api_key = os.getenv("TOGETHER_API_KEY")
    print("\n🔑 Estado de la API Key:", "Cargada correctamente" if api_key else "No encontrada")
    
    if not api_key:
        raise ValueError("La variable 'TOGETHER_API_KEY' no está configurada.")
    return api_key

def listar_modelos_vision(client):
    """Lista los modelos de visión disponibles"""
    print("\n🔍 Buscando modelos de visión disponibles...")
    models = client.models.list()
    vision_models = [model.id for model in models if "Vision" in model.id]
    
    print("\n🖼️ Modelos de visión disponibles:")
    for model in vision_models:
        print(f"- {model}")
    
    return vision_models

def inicializar_cliente():
    """Configura y devuelve el cliente de Together"""
    api_key = verificar_api_key()
    client = Together(api_key=api_key)
    
    # Verificar modelos disponibles
    modelos_disponibles = listar_modelos_vision(client)
    if MODEL_NAME not in modelos_disponibles:
        print(f"\n⚠️ Advertencia: El modelo seleccionado '{MODEL_NAME}' no está en la lista de modelos disponibles")
    
    return client

def buscar_imagenes(directorio):
    """Recursivamente busca imágenes en subdirectorios"""
    formatos_validos = ('.png', '.jpg', '.jpeg', '.webp')
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directorio)
        for file in files if file.lower().endswith(formatos_validos)
    ]

def procesar_imagen(client, image_path):
    """Envía imagen al modelo y devuelve el texto extraído"""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_INSTRUCCIONES},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
            stream=False,
            max_tokens=50
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"\n⚠️ Error procesando {os.path.basename(image_path)}: {str(e)}")
        return "ERROR"

def main():
    print("\n" + "="*50)
    print("=== Sistema de Extracción de Nombres de Medicamentos ===")
    print("="*50)

    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Inicializar componentes
    client = inicializar_cliente()
    
    # Verificar existencia del directorio
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"\n❌ Directorio no encontrado: {DATASET_PATH}")

    todas_imagenes = buscar_imagenes(DATASET_PATH)
    
    if not todas_imagenes:
        raise ValueError("\n❌ No se encontraron imágenes en el directorio especificado")

    # Selección aleatoria controlada
    random.seed(42)  # Para reproducibilidad
    imagenes_procesar = random.sample(todas_imagenes, min(SAMPLE_SIZE, len(todas_imagenes)))
    
    # Preparar resultados
    resultados = []
    print(f"\n🔍 Procesando {len(imagenes_procesar)} imágenes con el modelo '{MODEL_NAME}'...")

    # Procesamiento con barra de progreso
    for img_path in tqdm(imagenes_procesar, desc="Progreso"):
        nombre_archivo = os.path.basename(img_path)
        nombre_extraido = procesar_imagen(client, img_path)
        
        resultados.append({
            "Archivo": nombre_archivo,
            "Ruta Completa": img_path,
            "Medicamento Detectado": nombre_extraido
        })

    # Guardar y mostrar resultados
    df = pd.DataFrame(resultados)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"✅ Proceso completado exitosamente!")
    print(f"📊 Resultados guardados en: {OUTPUT_CSV}")
    print("\nResumen estadístico:")
    print(df["Medicamento Detectado"].value_counts())
    print("="*50)

if __name__ == "__main__":
    main()
