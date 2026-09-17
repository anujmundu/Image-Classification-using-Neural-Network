# Use official Python image
FROM python:3.12-slim as base

# Set workdir
WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .
COPY api/requirements.txt api_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r api_requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY model.py ./src/model.py
COPY data_loader.py ./src/data_loader.py
COPY predict.py ./src/predict.py

# Expose port for FastAPI
EXPOSE 8000

# Set entrypoint
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
