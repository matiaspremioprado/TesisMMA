FROM public.ecr.aws/lambda/python:3.11

# Instalar dependencias del sistema
RUN yum -y update && \
    yum -y install \
        python3 \
        python3-devel \
        gcc \
        gcc-c++ \
        zlib-devel \
        libjpeg-devel \
        tesseract \
        tesseract-langpack-spa \
        leptonica \
        make && \
    yum clean all

# Asegurar que Tesseract esté en el PATH
ENV PATH="/usr/bin:${PATH}"
ENV TESSDATA_PREFIX="/usr/share/tessdata"

# Copiar archivos de la aplicación
COPY app/main.py ./main.py
COPY app/requirements.txt ./requirements.txt
COPY app/workspace/resources/credentials/ocr_credentials.json ./workspace/resources/credentials/ocr_credentials.json

# Instalar dependencias de Python
RUN pip install --upgrade pip wheel setuptools
RUN pip install --only-binary=:all: pyarrow==14.0.2
RUN pip install --prefer-binary numpy && \
    pip install -r requirements.txt

# Comando para ejecutar Lambda
CMD ["main.lambda_handler"]

RUN ls -lR /var/task/workspace

