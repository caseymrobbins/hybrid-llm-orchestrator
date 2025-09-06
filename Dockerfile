# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY..

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]