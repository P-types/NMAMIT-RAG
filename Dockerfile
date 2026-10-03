FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Hugging Face cache location
ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/opt/huggingface

# --------------------------------------------------
# System dependencies
# --------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python dependencies
# --------------------------------------------------

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# --------------------------------------------------
# Application
# --------------------------------------------------

COPY api.py .
COPY graph_rag.py .

# --------------------------------------------------
# Download the SAME embedding model used by graph_rag.py
# --------------------------------------------------

RUN python -c "\
import graph_rag; \
from sentence_transformers import SentenceTransformer; \
print('Downloading embedding model:', graph_rag.EMBEDDING_MODEL); \
SentenceTransformer(graph_rag.EMBEDDING_MODEL); \
print('Embedding model cached successfully') \
"

# --------------------------------------------------
# Runtime
# --------------------------------------------------

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]