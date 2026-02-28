# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - tesseract-ocr: OCR engine
# - tesseract-ocr-eng: English language data for Tesseract
# - poppler-utils: PDF utilities (required by pdf2image)
# - libgl1-mesa-glx: OpenGL libraries (required by some image processing)
# - libglib2.0-0: Required by OpenCV if you add it later

# libgl1 and libglx-mesa0 instead is needed 
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libglx-mesa0 \
    libgl1 \
    libglib2.0-0 \
    curl  \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create data directories
RUN mkdir -p /data/input /data/pngs /data/processing /data/processed /data/failed

# Copy shared code (used by all services)
COPY shared/ /app/shared/

# Copy AI processor code (Ollama client, etc.)
COPY ai_processor/ /app/ai_processor/

# Copy service-specific code
COPY /pdf_processing/ /app/

# Set Python path to include /app for imports
ENV PYTHONPATH=/app

# Health check (optional but recommended)
# Checks if the main.py file exists and Python is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run the application
CMD ["python", "main.py"]
