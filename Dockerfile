# specialized Dockerfile for Hugging Face Spaces (CPU only)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Force CPU only for torch to save space
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install system dependencies
# - libgl1: for opencv
# - tesseract-ocr: for OCR fallback
# - gcc: for building some python deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to use cache
COPY requirements.txt .

# Install python dependencies
# explicitly install CPU versions of torch first to avoid downloading huge GPU wheels
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create a user to run the app (security best practice)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

CMD ["uvicorn", "shelfsense.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
