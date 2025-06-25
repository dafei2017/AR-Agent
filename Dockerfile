# Multi-stage Dockerfile for AR-Agent
# Medical Multimodal Augmented Reality Agent

# Base stage with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3.9-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install -r requirements.txt

# Install development dependencies
RUN pip install pytest pytest-cov black isort flake8 mypy pre-commit

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data/images /app/data/results /app/data/models /app/data/cache /app/logs /app/uploads

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 5000

# Development command
CMD ["python", "app.py"]

# Production stage
FROM base as production

# Install production dependencies only
WORKDIR /app

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install production dependencies
RUN pip install -r requirements.txt gunicorn

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY templates/ ./templates/
COPY static/ ./static/
COPY app.py ./

# Install the package
RUN pip install .

# Create non-root user for security
RUN groupadd -r aruser && useradd -r -g aruser aruser

# Create necessary directories and set ownership
RUN mkdir -p /app/data/images /app/data/results /app/data/models /app/data/cache /app/logs /app/uploads \
    && chown -R aruser:aruser /app

# Switch to non-root user
USER aruser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "app:app"]

# Minimal stage for inference only
FROM base as minimal

WORKDIR /app

# Install minimal dependencies
COPY requirements.txt ./
RUN pip install torch torchvision transformers accelerate peft bitsandbytes \
    opencv-python pillow numpy flask flask-cors pyyaml requests tqdm

# Copy only necessary files
COPY src/medical_analyzer/ ./src/medical_analyzer/
COPY configs/config.yaml ./configs/
COPY app.py ./

# Create minimal user
RUN groupadd -r aruser && useradd -r -g aruser aruser
RUN mkdir -p /app/data /app/logs && chown -R aruser:aruser /app
USER aruser

EXPOSE 5000
CMD ["python", "app.py"]

# GPU-optimized stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-optimized

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libopencv-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install optimized PyTorch for CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy and install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application
COPY . .
RUN pip install .

# Setup user
RUN groupadd -r aruser && useradd -r -g aruser aruser
RUN mkdir -p /app/data /app/logs && chown -R aruser:aruser /app
USER aruser

EXPOSE 5000
CMD ["python", "app.py"]

# Default target is production
FROM production