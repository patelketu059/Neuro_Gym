# ── Neuro Gym RAG — HuggingFace Spaces Dockerfile ────────────────────────────
# Runtime: CPU-only (embeddings via OpenRouter API, no local GPU model)
# Port:    7860 (HF Spaces standard)

FROM python:3.11-slim

# System deps — build tools for packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces runs as a non-root user — ensure /tmp is writable for artifact cache
RUN chmod -R 777 /tmp

# BM25 artifacts are downloaded to /tmp/gym-rag-artifacts at first boot.
# Set ARTIFACT_DIR to a persistent volume path if available.
ENV ARTIFACT_DIR=/tmp/gym-rag-artifacts

# HF Spaces standard port
EXPOSE 7860

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
