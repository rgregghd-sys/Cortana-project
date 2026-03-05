FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (no models/, no .env, no chroma_db — see .dockerignore)
COPY cortana/ ./cortana/
COPY app.py .

# Data directory for SQLite + ChromaDB on persistent volume
RUN mkdir -p /data /app/cortana/static
ENV CORTANA_DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
ENV LLAMA_ENABLED=false

EXPOSE 8080

CMD ["python", "-m", "cortana.main", "--web", "--host", "0.0.0.0", "--port", "8080"]
