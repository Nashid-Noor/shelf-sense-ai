# =============================================================================
# ShelfSense AI - Production Dockerfile
# =============================================================================
# Multi-stage build optimized for:
# - Small final image size
# - GPU support (NVIDIA CUDA)
# - Fast builds with layer caching
# - Security best practices
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Python with CUDA
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    # Image processing
    libpng-dev \
    libjpeg-dev \
    libwebp-dev \
    libtiff-dev \
    # OCR dependencies (Tesseract)
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    # Build tools
    build-essential \
    pkg-config \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for Python 3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create non-root user for security
RUN groupadd --gid 1000 shelfsense \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash shelfsense

# Set working directory
WORKDIR /app


# -----------------------------------------------------------------------------
# Stage 2: Dependencies Builder
# -----------------------------------------------------------------------------
FROM base AS builder

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Pre-download models to include in image (optional, uncomment to enable)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# RUN python -c "import easyocr; easyocr.Reader(['en'])"


# -----------------------------------------------------------------------------
# Stage 3: Production Image
# -----------------------------------------------------------------------------
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=shelfsense:shelfsense . /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/uploads \
    && chown -R shelfsense:shelfsense /app

# Set environment variables
ENV SHELFSENSE_ENV=production \
    SHELFSENSE_LOG_LEVEL=INFO \
    SHELFSENSE_DATA_DIR=/app/data \
    SHELFSENSE_MODEL_DIR=/app/models \
    SHELFSENSE_UPLOAD_DIR=/app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Switch to non-root user
USER shelfsense

# Default command
CMD ["python", "-m", "uvicorn", "shelfsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# -----------------------------------------------------------------------------
# Stage 4: Development Image
# -----------------------------------------------------------------------------
FROM production AS development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional dev tools
RUN pip install --no-cache-dir \
    debugpy \
    watchdog[watchmedo]

# Enable hot reload
ENV SHELFSENSE_ENV=development \
    SHELFSENSE_DEBUG=true \
    SHELFSENSE_LOG_LEVEL=DEBUG

# Override command for development (with reload)
CMD ["python", "-m", "uvicorn", "shelfsense.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# -----------------------------------------------------------------------------
# Stage 5: CPU-only Image (smaller, for non-GPU environments)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS cpu-only

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 shelfsense \
    && useradd --uid 1000 --gid 1000 --create-home shelfsense

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY --chown=shelfsense:shelfsense . /app

# Create directories
RUN mkdir -p /app/data /app/models /app/logs /app/uploads \
    && chown -R shelfsense:shelfsense /app

ENV SHELFSENSE_ENV=production \
    SHELFSENSE_DEVICE=cpu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER shelfsense

CMD ["python", "-m", "uvicorn", "shelfsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
