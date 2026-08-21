FROM python:3.11-slim

# Install ffmpeg for Sarvam voice STT audio processing and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Pre-fetch the fastembed ONNX model at build time. Without this the first
# request after every cold start pays for it at runtime instead (measured
# ~30s downloading from Hugging Face unauthenticated) -- bad on a platform
# that stops idle machines, since every wake-up would eat that cost again.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Expose default ports
EXPOSE 7860 10000

# Start Uvicorn dynamically binding to $PORT
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
