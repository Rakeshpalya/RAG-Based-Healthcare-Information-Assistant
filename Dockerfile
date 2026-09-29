# ==============================================================================
# AI-Healthcare-Agent: Multi-Stage Production Dockerfile
# Base: Python 3.11 Slim (Debian Bookworm)
# Security: Non-root execution (appuser:10001), no baked secrets
# Invariant: 744 FAISS vectors == 744 metadata records (Strict Read-Only)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install system build tools and dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache SentenceTransformer model weights for offline runtime
ENV HF_HOME=/opt/huggingface_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ------------------------------------------------------------------------------
# Stage 2: Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="HealthAI Engineering Team" \
      version="1.0.0" \
      description="Production Container for AI-Healthcare-Agent Platform"

WORKDIR /app

# Install minimal runtime system libraries:
# - libgomp1: required for FAISS OpenMP CPU acceleration
# - curl: required for Docker container native health check
# - ca-certificates: required for secure outbound SSL/TLS connections
RUN apt-get update && apt-get install --no-install-recommends -y \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and pre-cached model from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/huggingface_cache /app/cache/huggingface

# Environment configuration
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_OFFLINE=1

# Create non-root system user and group (UID/GID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application source tree and persistent vector store assets
COPY backend/ /app/backend/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini
COPY data/vector_store/ /app/data/vector_store/

# Create runtime directories and set secure ownership
RUN mkdir -p /app/data /app/evaluation_reports /app/logs /tmp && \
    chown -R appuser:appgroup /app /tmp

# Switch to non-root user
USER appuser

# Expose API service port
EXPOSE 8000

# Container healthcheck using /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production startup command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
