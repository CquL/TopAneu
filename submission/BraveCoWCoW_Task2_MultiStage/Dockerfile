FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app:/opt/app/nnXNet \
    nnXNet_compile=false \
    TOPANEU_MODEL_DIR=/opt/ml/model

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

COPY requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color -r /opt/app/requirements.txt

COPY nnXNet /opt/app/nnXNet
RUN python -m pip install --no-cache-dir --no-deps /opt/app/nnXNet

COPY main.py inference.py /opt/app/
COPY topaneu /opt/app/topaneu
ENV nnXNet_raw=/tmp/nnXNet_raw \
    nnXNet_preprocessed=/tmp/nnXNet_preprocessed \
    nnXNet_results=/tmp/nnXNet_results
RUN chown -R user:user /opt/app

USER user
ENTRYPOINT ["python", "/opt/app/main.py"]
