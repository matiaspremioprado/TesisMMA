import json
import boto3
import os
from PIL import Image
import pillow_heif

# Cliente de S3
s3 = boto3.client("s3")

OUTPUT_BUCKET = "medicamentos-output-tesismma"
FRIENDLY_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

def convert_heic_to_jpg(bucket_name, object_key):
    heic_filename = os.path.basename(object_key)
    local_heic_path = f"/tmp/{heic_filename}"
    s3.download_file(bucket_name, object_key, local_heic_path)
    
    heif_file = pillow_heif.open_heif(local_heic_path)
    image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data,
        "raw", heif_file.mode, heif_file.stride
    )
    
    jpg_filename = os.path.splitext(heic_filename)[0] + ".jpg"
    local_jpg_path = f"/tmp/{jpg_filename}"
    image.save(local_jpg_path, "JPEG", quality=95)

    new_s3_key = f"convertidas/{jpg_filename}"
    s3.upload_file(local_jpg_path, bucket_name, new_s3_key, ExtraArgs={'ContentType': 'image/jpeg'})
    s3.upload_file(local_jpg_path, OUTPUT_BUCKET, new_s3_key, ExtraArgs={'ContentType': 'image/jpeg'})

    return new_s3_key

def move_friendly_image(bucket_name, object_key):
    filename = os.path.basename(object_key)
    local_path = f"/tmp/{filename}"
    s3.download_file(bucket_name, object_key, local_path)

    new_s3_key = f"convertidas/{filename}"
    s3.upload_file(local_path, bucket_name, new_s3_key)
    s3.upload_file(local_path, OUTPUT_BUCKET, new_s3_key)

    return new_s3_key

def lambda_handler(event, context):
    try:
        record = event["Records"][0]
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = record["s3"]["object"]["key"]
        
        # Ignorar archivos dentro de 'convertidas/' para evitar loops
        if object_key.startswith("convertidas/"):
            return {
                "statusCode": 200,
                "body": json.dumps("Ignorado: archivo en carpeta convertidas.")
            }

        ext = os.path.splitext(object_key)[1].lower()
        
        if ext == ".heic":
            new_file = convert_heic_to_jpg(bucket_name, object_key)
            return {
                "statusCode": 200,
                "body": json.dumps(f"Conversión completada: {new_file}")
            }
        elif ext in FRIENDLY_EXTENSIONS:
            new_file = move_friendly_image(bucket_name, object_key)
            return {
                "statusCode": 200,
                "body": json.dumps(f"Archivo movido: {new_file}")
            }
        else:
            return {
                "statusCode": 400,
                "body": json.dumps(f"Formato no soportado: {ext}")
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }
