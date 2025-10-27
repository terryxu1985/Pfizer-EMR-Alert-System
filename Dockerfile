# Pfizer EMR Alert System Dockerfile
# Multi-stage build for production-ready containerization

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Build arguments for version tracking and metadata
ARG VERSION=2.2.0
ARG BUILD_DATE
ARG VCS_REF
ARG BUILD_TYPE=production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_VERSION=${VERSION} \
    BUILD_DATE=${BUILD_DATE} \
    VCS_REF=${VCS_REF} \
    BUILD_TYPE=${BUILD_TYPE}

# Add Docker labels for image metadata
LABEL version="${VERSION}" \
      maintainer="Pfizer EMR Development Team" \
      org.opencontainers.image.title="Pfizer EMR Alert System" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.description="Enterprise AI-Powered Clinical Decision Support System" \
      org.opencontainers.image.vendor="Pfizer" \
      com.pfizer.app.version="${VERSION}" \
      com.pfizer.build.type="${BUILD_TYPE}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements files
COPY config/requirements_production.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create necessary directories with proper ownership
RUN mkdir -p /app/logs /app/data/storage /app/data/model_ready && \
    chown -R appuser:appuser /app

# Set permissions
RUN chmod +x /app/scripts/startup/run_*.py && \
    chmod +x /app/frontend/server/emr_ui_server.py

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8080

# Health check for API service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "scripts/startup/run_complete_system.py"]

