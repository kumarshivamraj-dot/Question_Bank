FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libglx-mesa0 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/uploads /app/data/processed

COPY study_pipeline/ /app/study_pipeline/
COPY study_web/ /app/study_web/
COPY ai_processor/ /app/ai_processor/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "study_web.app:app", "--host", "0.0.0.0", "--port", "8000"]
