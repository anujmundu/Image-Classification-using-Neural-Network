# Multi-stage production container for 11-Architecture Vision Benchmark API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/classes.json /app/models/
COPY models/model_efficientnet_b4_latest.pth /app/models/
COPY models/model_mobilenet_v3_large_latest.pth /app/models/
COPY models/model_convnext_tiny_latest.pth /app/models/
COPY models/model_resnet101_latest.pth /app/models/
COPY models/model_efficientnet_b4_fp32.onnx /app/models/
COPY models/model_efficientnet_b4_int8.onnx /app/models/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Launch production ASGI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
