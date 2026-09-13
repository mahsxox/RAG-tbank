FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    DATA_DIR=/app/data \
    QDRANT_URL=http://qdrant:6333

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY src ./src
COPY db_scripts ./db_scripts
COPY app.py .
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /app/data /models

EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ui"]
