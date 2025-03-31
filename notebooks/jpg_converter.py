import os
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".heic"):
                heic_path = os.path.join(subdir, file)
                jpg_path = os.path.join(subdir, os.path.splitext(file)[0] + ".jpg")
                
                # Leer el archivo HEIC y convertirlo a JPG
                heif_file = pillow_heif.open_heif(heic_path)
                image = Image.frombytes(
                    heif_file.mode, heif_file.size, heif_file.data,
                    "raw", heif_file.mode, heif_file.stride
                )
                image.save(jpg_path, "JPEG", quality=95)
                
                # Eliminar el archivo HEIC original
                os.remove(heic_path)
                print(f"Convertido: {heic_path} -> {jpg_path}")

if __name__ == "__main__":
    dataset_path = "workspace/resources/datasets/dataset_medicamentos"
    convert_heic_to_jpg(dataset_path)
    print("Conversión completada.")
