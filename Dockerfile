FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git git-lfs \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models_cache data/chromadb templates static

# HuggingFace Spaces expects port 7860
EXPOSE 7860

ENV EDEN_PORT=7860
ENV EDEN_HOST=0.0.0.0
ENV EDEN_HARDWARE_PROFILE=auto
ENV EDEN_MODELS_CACHE=/app/models_cache
ENV EDEN_LOG_LEVEL=INFO

CMD ["python3", "app.py"]
