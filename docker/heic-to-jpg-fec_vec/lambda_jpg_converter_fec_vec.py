import json
import boto3
import os
from PIL import Image
import pillow_heif

# Cliente de S3
s3 = boto3.client("s3")

# Bucket de salida
OUTPUT_BUCKET = "medicamentos-output-tesismma"

def convert_heic_to_jpg(bucket_name, object_key):
    heic_filename = os.path.basename(object_key)
    local_heic_path = f"/tmp/{heic_filename}"
    s3.download_file(bucket_name, object_key, local_heic_path)
    
    heif_file = pillow_heif.open_heif(local_heic_path)
    image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data,
        "raw", heif_file.mode, heif_file.stride
    )

    base_name = os.path.splitext(heic_filename)[0]
    new_name = f"{base_name}-fec-vec.jpg"
    local_jpg_path = f"/tmp/{new_name}"
    image.save(local_jpg_path, "JPEG", quality=95)

    new_s3_key = f"convertidas/{new_name}"
    s3.upload_file(local_jpg_path, bucket_name, new_s3_key, ExtraArgs={'ContentType': 'image/jpeg'})
    s3.upload_file(local_jpg_path, OUTPUT_BUCKET, new_s3_key, ExtraArgs={'ContentType': 'image/jpeg'})

    return new_s3_key

def process_jpg(bucket_name, object_key):
    jpg_filename = os.path.basename(object_key)
    local_jpg_path = f"/tmp/{jpg_filename}"
    s3.download_file(bucket_name, object_key, local_jpg_path)

    base_name = os.path.splitext(jpg_filename)[0]
    new_name = f"{base_name}-fec-vec.jpg"
    new_s3_key = f"convertidas/{new_name}"

    s3.upload_file(local_jpg_path, bucket_name, new_s3_key, ExtraArgs={'ContentType': 'image/jpeg'})
    s3.upload_file(local_jpg_path, OUTPUT_BUCKET, new_s3_key, ExtraArgs={'ContentType': 'image/jpeg'})

    return new_s3_key

def lambda_handler(event, context):
    try:
        record = event["Records"][0]
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = record["s3"]["object"]["key"]

        # ✅ Evita loops si ya fue procesado
        if object_key.startswith("convertidas/"):
            print(f"Ignorado: archivo ya procesado ({object_key})")
            return {
                "statusCode": 200,
                "body": json.dumps("Archivo ya procesado, no se ejecuta nuevamente.")
            }

        ext = os.path.splitext(object_key)[1].lower()

        if ext == ".heic":
            new_file = convert_heic_to_jpg(bucket_name, object_key)
            return {
                "statusCode": 200,
                "body": json.dumps(f"Conversión HEIC completada: {new_file}")
            }
        elif ext in [".jpg", ".jpeg"]:
            new_file = process_jpg(bucket_name, object_key)
            return {
                "statusCode": 200,
                "body": json.dumps(f"Procesamiento JPG completado: {new_file}")
            }
        else:
            return {
                "statusCode": 400,
                "body": json.dumps("Formato de archivo no soportado (solo HEIC/JPG)")
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }
