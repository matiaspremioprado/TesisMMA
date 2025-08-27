#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lambda OCR pipeline (COMBINADO)
--------------------------------

Requisitos del usuario:
- Si la imagen **NO** termina con "-fec-vec" (antes de la extensión), ejecutar **exactamente** el flujo del PRIMER script (medicamento):
  * Extraer texto con Together Vision (stream)
  * Normalizar
  * Regla ULTRADIM
  * Matching con diccionario (Levenshtein + overlap)
  * Subir CSV a `DICCIONARIO_BUCKET` en `RESULTS_PREFIX`

- Si la imagen **SÍ** termina con "-fec-vec" (antes de la extensión), ejecutar el flujo de FECHA DE VENCIMIENTO basado en el SEGUNDO script,
  pero **mejorado** con la lógica del TERCER script (Colab):
  * Prompt para fechas
  * Hasta 5 intentos de OCR (streams independientes)
  * Extracción robusta de MM/AAAA (con fallback a DD/MM/AAAA si está disponible)
  * Actualizar el último CSV en `RESULTS_PREFIX` seteando la columna "Fecha de vencimiento"

NOTA: El flujo de MEDICAMENTO se mantiene sin cambios funcionales respecto al primer archivo.
"""
import os
import json
import re
import base64
import tempfile
import boto3
import pandas as pd
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from unidecode import unidecode

# Together SDK (igual que en el primer script)
try:
    from together import Together
except Exception:
    Together = None

# Levenshtein (igual que en el primer script)
try:
    import Levenshtein
    _has_lev = True
except Exception:
    import difflib
    _has_lev = False

# =============================
# CONFIGURACIÓN (sin cambios)
# =============================
MODEL_NAME = os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo")
DICCIONARIO_BUCKET = os.environ.get("DICCIONARIO_BUCKET", "medicamentos-output-tesismma")
DICCIONARIO_KEY = os.environ.get("DICCIONARIO_KEY", "diccionarios/diccionario_medicamentos.csv")
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "resultados/")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))  # Para flujo MEDICAMENTO
RETRY_BACKOFF_BASE = float(os.environ.get("RETRY_BACKOFF_BASE", "1.6"))
TOGETHER_API_PATH = Path(os.environ.get("OCR_CREDENTIALS_PATH", "/var/task/workspace/resources/credentials/ocr_credentials.json"))

# Prompt del flujo MEDICAMENTO (igual al primer script)
getDescriptionPrompt = (
    "Extract all visible text from the image exactly as written, do NOT generate descriptions, "
    "interpretations, or summaries. Only output the raw text found in the image, maintaining original "
    "formatting (e.g., line breaks, spacing). If no text is detectable, reply with 'No visible text found.'"
)

# Prompt del flujo FECHA (tomado del Colab, manteniendo intención)
getDatePrompt = (
    "Extract all visible text from the image exactly as written, do NOT generate descriptions, "
    "interpretations, or summaries. Only output the raw text found in the image, maintaining original "
    "formatting (e.g., line breaks, spacing). If no text is detectable, reply with 'No visible text found.'"
    "You should be finding dates."
)

# Logging (igual estilo que el primer script)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr_lambda")

# =============================
# HELPERS COMUNES
# =============================

def verificar_api_key():
    """Obtiene la API key de Together desde env var o archivo JSON (mismo comportamiento que el primer script)."""
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        logger.info("TOGETHER_API_KEY encontrada en variable de entorno.")
        return key
    if TOGETHER_API_PATH.exists():
        try:
            with open(TOGETHER_API_PATH, "r", encoding="utf-8") as f:
                creds = json.load(f)
            key = creds.get("TOGETHER_API_KEY")
            if key:
                logger.info(f"TOGETHER_API_KEY leída desde {TOGETHER_API_PATH}")
                return key
            else:
                raise RuntimeError("TOGETHER_API_KEY no presente en archivo JSON.")
        except Exception as e:
            raise RuntimeError(f"Error leyendo credenciales: {e}")
    raise RuntimeError("TOGETHER_API_KEY no encontrada. Define variable de entorno o fichero de credenciales.")


def crear_cliente_together():
    api_key = verificar_api_key()
    os.environ["TOGETHER_API_KEY"] = api_key
    if Together is None:
        raise RuntimeError("SDK de Together no disponible. Instalar paquete correcto en requirements.")
    return Together()


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# =============================
# FLUJO MEDICAMENTO (SIN CAMBIOS)
# =============================

def procesar_imagen_stream(client, image_path, prompt, max_retries=MAX_RETRIES):
    """Igual que en el primer script: hace streaming con retry/backoff y concatena el contenido."""
    attempt = 0
    last_exc = None
    while attempt < max_retries:
        try:
            base64_image = encode_image(image_path)
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }],
                stream=True,
            )
            resultado = ""
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content = choice.delta.content
                        if content is not None:
                            resultado += content
            return resultado.strip()
        except Exception as e:
            last_exc = e
            logger.warning(f"Intento {attempt+1}/{max_retries} falló: {e}")
            sleep_for = (RETRY_BACKOFF_BASE ** attempt) + (0.1 * attempt)
            time.sleep(min(sleep_for, 30))
            attempt += 1
    raise RuntimeError(f"Fallo tras {max_retries} intentos. Último error: {last_exc}")


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unidecode(text).upper()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_score(a, b):
    if not a and not b:
        return 1.0
    if _has_lev:
        dist = Levenshtein.distance(a, b)
    else:
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        dist = max(0.0, (1.0 - ratio) * max(len(a), len(b), 1))
    return 1.0 / (1.0 + dist)


def cargar_diccionario_desde_s3(bucket_name, diccionario_key):
    s3 = boto3.client("s3")
    logger.info(f"Descargando diccionario desde s3://{bucket_name}/{diccionario_key}")
    obj = s3.get_object(Bucket=bucket_name, Key=diccionario_key)
    df = pd.read_csv(obj["Body"], dtype=str).fillna("")
    if "Input" not in df.columns:
        raise RuntimeError("El diccionario no contiene la columna 'Input'.")
    df["Input"] = df["Input"].apply(normalize_text)
    return df


def find_medication_info(extracted_text, medication_df):
    extracted_text_norm = normalize_text(extracted_text)
    extracted_words = set(extracted_text_norm.split())

    exact_matches = medication_df[medication_df["Input"] == extracted_text_norm]
    if not exact_matches.empty:
        row = exact_matches.iloc[0]
        return row.get("Nombre del medicamento", ""), row.get("Dosis", "")

    best_match_name = ""
    best_match_dose = ""
    best_score = float("-inf")

    for _, row in medication_df.iterrows():
        candidate = row["Input"]
        candidate_words = set(candidate.split())

        if extracted_text_norm == candidate:
            return row.get("Nombre del medicamento", ""), row.get("Dosis", "")

        word_overlap_score = len(extracted_words.intersection(candidate_words))
        lev_score = levenshtein_score(extracted_text_norm, candidate)
        combined_score = word_overlap_score * 1.5 + lev_score * 5.0

        if combined_score > best_score:
            best_score = combined_score
            best_match_name = row.get("Nombre del medicamento", "")
            best_match_dose = row.get("Dosis", "")

    if best_score > 1:
        return best_match_name, best_match_dose
    else:
        return "No encontrado", ""


def upload_result_csv(bucket, base_name, df_result):
    resultado_path = "/tmp/resultado.csv"
    df_result.to_csv(resultado_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    s3_key_out = f"{RESULTS_PREFIX}{base_name}_{timestamp}.csv"
    s3 = boto3.client("s3")
    s3.upload_file(Filename=resultado_path, Bucket=bucket, Key=s3_key_out)
    logger.info(f"Resultado subido a s3://{bucket}/{s3_key_out}")
    return s3_key_out


# =============================
# FLUJO FECHA DE VENCIMIENTO (MEJORADO)
# =============================

def is_fec_vec_key(key: str) -> bool:
    """Detecta si el objeto corresponde a una imagen de fecha de vencimiento por sufijo '-fec-vec'."""
    base, ext = os.path.splitext(key.lower())
    if ext not in [".jpg", ".jpeg", ".png"]:
        return False
    return base.endswith("-fec-vec")


def procesar_imagen_stream_once(client, image_path, prompt):
    """Un solo stream (sin backoff). Se usa para múltiples intentos controlados en fec-vec."""
    try:
        base64_image = encode_image(image_path)
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
            stream=True,
        )
        resultado = ""
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    content = choice.delta.content
                    if content is not None:
                        resultado += content
        return resultado.strip()
    except Exception as e:
        logger.warning(f"OCR (una vez) falló: {e}")
        return ""


def mes_a_numero(mes_str: str) -> int:
    meses = {
        "ene": 1, "feb": 2, "mar": 3, "abr": 4,
        "may": 5, "jun": 6, "jul": 7, "ago": 8,
        "sep": 9, "oct": 10, "nov": 11, "dic": 12,
        "jan": 1, "apr": 4, "aug": 8, "dec": 12,
    }
    key = (mes_str or "")[:3].lower()
    return meses.get(key, 0)


def extract_mm_yyyy_improved(text: str) -> str:
    """Port de Colab: busca patrones MM/AAAA o MES AAAA y retorna 'MM_YYYY' o 'No encontrada'."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    txt = text.lower().replace("\n", " ").replace("\r", " ")
    txt = re.sub(r"[^\w\s/\\\-–—.:]", "", txt)

    current_year = datetime.now().year
    min_valid_year = current_year - 10
    max_valid_year = current_year + 10

    patterns = [
        r"(0[1-9]|1[0-2])[/\\\-–—](20\d{2})",
        r"(0[1-9]|1[0-2])[/\\\-–—](\d{2})(?!\d)",
        r"(ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-–—/\\]*(\d{4})",
        r"(\d{1,2})[/\\\-–—](0[1-9]|1[0-2])[/\\\-–—](20\d{2})",
        r"(20\d{2})[\-–—/\\](0[1-9]|1[0-2])",
        r"(0[1-9]|1[0-2])(20\d{2})",
        r"(0[1-9]|1[0-2])(\d{2})(?!\d)",
    ]

    valid_dates = []
    for pattern in patterns:
        for match in re.finditer(pattern, txt, re.IGNORECASE):
            groups = match.groups()
            try:
                if len(groups) == 2:
                    month, year = groups
                    if len(year) == 2:
                        year = f"20{year}"
                    year_num = int(year)
                    month_num = mes_a_numero(month) if month.isalpha() else int(month)
                    if min_valid_year <= year_num <= max_valid_year and 1 <= month_num <= 12:
                        valid_dates.append((month_num, year_num))
                elif len(groups) == 3:
                    day, month, year = groups
                    year_num = int(year)
                    month_num = int(month)
                    if min_valid_year <= year_num <= max_valid_year and 1 <= month_num <= 12:
                        valid_dates.append((month_num, year_num))
            except (ValueError, TypeError):
                continue

    if valid_dates:
        # Elegir la más reciente
        valid_dates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        best_month, best_year = valid_dates[0]
        return f"{best_month:02d}_{best_year}"
    return "No encontrada"


def extract_dd_mm_yyyy(text: str) -> str:
    """Extrae la primera coincidencia DD/MM/YYYY si existe."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    m = re.search(r"\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(20\d{2})\b", text)
    return m.group(0) if m else ""


def get_latest_csv_key(bucket: str, prefix: str = RESULTS_PREFIX) -> str:
    """Obtiene el último CSV (por LastModified) bajo `prefix` en `bucket` (maneja paginación)."""
    s3 = boto3.client("s3")
    continuation_token = None
    newest = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if not key.lower().endswith(".csv"):
                continue
            if (newest is None) or (obj["LastModified"] > newest["LastModified"]):
                newest = obj
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    if not newest:
        raise RuntimeError("No hay archivos CSV en la carpeta de resultados.")
    return newest["Key"]


def update_csv_expiry(bucket: str, key: str, expiry_value: str) -> None:
    """Actualiza/crea la columna 'Fecha de vencimiento' en el CSV y setea el valor en la fila 0."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"], dtype=str).fillna("")

    # Asegurar columna por nombre (mejor que por índice)
    if "Fecha de vencimiento" not in df.columns:
        df["Fecha de vencimiento"] = ""

    if df.shape[0] < 1:
        # Si por alguna razón está vacío, creamos una fila
        df.loc[0, :] = ""

    df.loc[0, "Fecha de vencimiento"] = expiry_value

    tmp_out = "/tmp/actualizado.csv"
    df.to_csv(tmp_out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    s3.upload_file(Filename=tmp_out, Bucket=bucket, Key=key)
    logger.info(f"CSV actualizado en s3://{bucket}/{key} (Fecha de vencimiento='{expiry_value}')")


def fec_vec_flow(client, image_path: str) -> dict:
    """Flujo mejorado para imágenes '-fec-vec': múltiple OCR + extracción robusta.

    Retorna un dict con: {
        'fecha_obtenida': <str>,
        'ocr_text': <str>,
        'intentos': <int>,
        'csv_actualizado_key': <str or None>
    }
    """
    ocr_text = ""
    fecha_full = ""  # DD/MM/YYYY si existe
    fecha_mm_yyyy = "No encontrada"

    intentos = 0
    for attempt in range(5):  # Igual que Colab: hasta 5 intentos
        intentos = attempt + 1
        raw = procesar_imagen_stream_once(client, image_path, getDatePrompt)
        if raw:
            ocr_text = raw
            # Primero, intentar fecha completa DD/MM/YYYY
            fecha_full = extract_dd_mm_yyyy(raw)
            # Luego, mes/año robusto
            fecha_mm_yyyy = extract_mm_yyyy_improved(raw)
            if fecha_full or (fecha_mm_yyyy and fecha_mm_yyyy != "No encontrada"):
                break

    # Decidir qué escribir en el CSV (preferimos DD/MM/YYYY; sino MM/YYYY)
    if fecha_full:
        fecha_para_csv = fecha_full
    elif fecha_mm_yyyy and fecha_mm_yyyy != "No encontrada":
        # Convertir "MM_YYYY" a "MM/YYYY" para mayor compatibilidad
        mm, yyyy = fecha_mm_yyyy.split("_")
        fecha_para_csv = f"{mm}/{yyyy}"
    else:
        # Mantener una marca clara si no hay texto visible
        if not ocr_text or ocr_text.strip().lower().startswith("no visible text"):
            fecha_para_csv = "No visible text found"
        else:
            fecha_para_csv = "No encontrada"

    # Actualizar el último CSV de resultados
    try:
        latest_key = get_latest_csv_key(DICCIONARIO_BUCKET, RESULTS_PREFIX)
        update_csv_expiry(DICCIONARIO_BUCKET, latest_key, fecha_para_csv)
    except Exception as e:
        logger.warning(f"No se pudo actualizar el último CSV: {e}")
        latest_key = None

    return {
        "fecha_obtenida": fecha_para_csv,
        "ocr_text": ocr_text,
        "intentos": intentos,
        "csv_actualizado_key": latest_key,
    }


# =============================
# LAMBDA HANDLER (COMBINADO)
# =============================

def lambda_handler(event, context):
    logger.info("Evento recibido")

    # --- Parseo de evento S3 ---
    try:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
    except Exception as e:
        logger.error(f"Evento no es S3: {e}")
        return {"statusCode": 400, "body": "Evento inválido"}

    # Ignorar eventos generados por la propia Lambda (igual que primer script)
    user_identity = record.get("userIdentity", {}).get("principalId", "")
    if "Lambda" in user_identity:
        logger.info("Evento generado por Lambda, ignorando")
        return {"statusCode": 200, "body": "Evento generado por Lambda ignorado"}

    # Solo procesar carpeta convertidas/
    if not key.startswith("convertidas/"):
        logger.info(f"Ignorando archivo fuera de convertidas/: {key}")
        return {"statusCode": 200, "body": "Archivo fuera de carpeta ignorado"}

    ext = os.path.splitext(key)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        logger.info(f"Ignorado archivo no imagen: {key}")
        return {"statusCode": 200, "body": f"Ignorado archivo no imagen: {key}"}

    # --- Descargar imagen temporalmente ---
    s3 = boto3.client("s3")
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            s3.download_fileobj(bucket, key, tf)
            tmp_file_path = tf.name
        logger.info(f"Imagen descargada a {tmp_file_path}")
    except Exception as e:
        logger.error(f"Error descargando imagen: {e}")
        return {"statusCode": 500, "body": f"Error descargando imagen: {e}"}

    # --- Cliente Together ---
    try:
        client = crear_cliente_together()
    except Exception as e:
        logger.error(f"No se pudo inicializar cliente Together: {e}")
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception:
                pass
        return {"statusCode": 500, "body": f"Error creando cliente: {e}"}

    # ======================================
    # BRANCH: FECHA DE VENCIMIENTO (fec-vec)
    # ======================================
    if is_fec_vec_key(key):
        logger.info("📆 Imagen reconocida como 'fecha de vencimiento' (sufijo -fec-vec)")
        try:
            result = fec_vec_flow(client, tmp_file_path)
            body = {
                "mensaje": "Fecha de vencimiento procesada",
                "fecha_obtenida": result["fecha_obtenida"],
                "intentos": result["intentos"],
                "s3_result_key": result["csv_actualizado_key"],
            }
            return {"statusCode": 200, "body": json.dumps(body)}
        except Exception as e:
            logger.exception(f"Error en flujo fec-vec: {e}")
            return {"statusCode": 500, "body": f"Error en flujo fec-vec: {e}"}
        finally:
            try:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
            except Exception:
                pass

    # =============================
    # BRANCH: MEDICAMENTO (SIN CAMBIOS)
    # =============================
    try:
        extracted_text = ""
        retry_count = 0
        nombre = "No encontrado"
        dosis = ""
        base_name = os.path.splitext(os.path.basename(key))[0]

        # Cargar diccionario una sola vez
        try:
            diccionario_medicamentos = cargar_diccionario_desde_s3(DICCIONARIO_BUCKET, DICCIONARIO_KEY)
        except Exception as e:
            logger.error(f"No se pudo cargar diccionario: {e}")
            return {"statusCode": 500, "body": f"No se pudo cargar diccionario: {e}"}

        df_out_row = {
            "Nombre Extraído": "",
            "Texto extraído": "",
            "Nombre Normalizado": "",
            "Nombre del medicamento": "",
            "Dosis": "",
            "Fecha de vencimiento": "",
        }

        while retry_count < MAX_RETRIES:
            try:
                raw_text = procesar_imagen_stream(client, tmp_file_path, getDescriptionPrompt)
            except Exception as e:
                logger.warning(f"Intento {retry_count+1} - error al procesar imagen: {e}")
                raw_text = ""
            raw_text_for_csv = (raw_text or "").replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()
            df_out_row["Nombre Extraído"] = raw_text_for_csv
            df_out_row["Texto extraído"] = (raw_text or "").strip().upper()
            df_out_row["Nombre Normalizado"] = normalize_text(raw_text)

            # Regla ULTRADIM (idéntica al primer script)
            if any(token in df_out_row["Texto extraído"] for token in ["ULTRADIM", "ULTRADIN", "ULTRA DIM", "ULTRA DIN"]):
                nombre = "Nopucid ULTRADIM"
                dosis = ""
                df_out_row["Nombre del medicamento"] = nombre
                df_out_row["Dosis"] = dosis
                logger.info("Regla ULTRADIM detectada. Resultado asignado.")
                break

            # Matching en diccionario
            nombre_match, dosis_match = find_medication_info(df_out_row["Nombre Normalizado"], diccionario_medicamentos)
            if nombre_match != "No encontrado":
                nombre = nombre_match
                dosis = dosis_match
                df_out_row["Nombre del medicamento"] = nombre
                df_out_row["Dosis"] = dosis
                logger.info(f"Matching exitoso en intento {retry_count+1}: {nombre} / {dosis}")
                break

            retry_count += 1
            logger.info(
                f"Intento {retry_count} completado - no match. Reintentando..." if retry_count < MAX_RETRIES else "Máximos reintentos alcanzados."
            )

        if nombre == "No encontrado":
            df_out_row["Nombre del medicamento"] = "No encontrado"
            df_out_row["Dosis"] = ""

        df_result = pd.DataFrame([df_out_row])
        s3_key_out = upload_result_csv(DICCIONARIO_BUCKET, base_name, df_result)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "nombre_extraido": df_out_row["Nombre Extraído"],
                "nombre_medicamento": df_out_row["Nombre del medicamento"],
                "dosis": df_out_row["Dosis"],
                "s3_result_key": s3_key_out,
            }),
        }

    except Exception as e:
        logger.exception(f"Error imprevisto en procesamiento: {e}")
        return {"statusCode": 500, "body": f"Error en procesamiento: {e}"}
    finally:
        try:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception:
            pass
