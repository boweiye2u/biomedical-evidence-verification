# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_ROOT=/artifacts \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY environments/serving-requirements.txt environments/serving-requirements.txt

RUN python -m pip install torch==2.7.0 --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -r environments/serving-requirements.txt

COPY configs/serving-6b-v1.json configs/serving-6b-v1.json
COPY retrieval/__init__.py retrieval/verification.py retrieval/
COPY serving/ serving/

EXPOSE 8000 8001

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
