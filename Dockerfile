# ==============================================================================
# Concept-Drift-Aware Fraud Detection Real-Time Microservice Dockerfile
# Multi-stage production build with NVIDIA CUDA GPU acceleration & Python 3.12
# ==============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv

WORKDIR /app

# Copy dependency definitions and install
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Copy application source code
COPY src/ /app/src/
COPY data/ /app/data/

# Expose microservice HTTP port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
