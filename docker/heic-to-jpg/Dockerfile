FROM public.ecr.aws/lambda/python:3.11

# Instalar librerías del sistema necesarias para pillow-heif
RUN yum install -y libheif libjpeg libpng libtiff && yum clean all

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el archivo principal
COPY lambda_jpg_converter.py .

# Indicar el handler de Lambda: archivo.función
CMD ["lambda_jpg_converter.lambda_handler"]
