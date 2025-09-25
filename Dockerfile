# Multi-Stage Dockerfile for Cultivate Learning ML MVP
# Issue #119: Production Docker Deployment Architecture
#
# Optimized for demo reliability, fast startup, and cost-effective Azure deployment
# Architecture supports lightweight production and full ML upgrade path

# =============================================================================
# Stage 1: Base Builder - Common foundation for all stages
# =============================================================================
FROM python:3.12-slim as builder

# Set build arguments for flexibility
ARG BUILD_ENV=production
ARG PYTHON_VERSION=3.12

# Set working directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Create non-root user for security (used in all stages)
RUN useradd -m -u 1000 apiuser

# =============================================================================
# Stage 2: Testing Stage - Includes E2E test dependencies
# =============================================================================
FROM builder as testing

LABEL stage=testing
LABEL purpose="E2E testing with Selenium support"

# Install testing dependencies including Selenium requirements
COPY requirements-api.txt requirements-testing.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-testing.txt

# Install Chrome for Selenium testing
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Copy all source code and tests
COPY src/ ./src/
COPY run_api.py test_demo_scenarios.py run_demo_tests.sh performance_test.sh ./
COPY requirements-*.txt ./

# Set ownership and switch to non-root user
RUN chown -R apiuser:apiuser /app
USER apiuser

# Command for running tests
CMD ["bash", "./run_demo_tests.sh", "--headless"]

# =============================================================================
# Stage 3: Production Stage - Lightweight for demo reliability
# =============================================================================
FROM python:3.12-slim as production

LABEL stage=production
LABEL purpose="Lightweight production deployment with heuristic ML models"
LABEL cost_optimization="optimized_for_demo"
LABEL architecture="lightweight_heuristics"

# Set working directory
WORKDIR /app

# Install minimal system dependencies (only curl for health checks)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install only production dependencies (lightweight)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt \
    && pip cache purge

# Copy application source code
COPY src/ ./src/
COPY run_api.py ./

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Enhanced health check for production reliability
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Optimized startup command with production settings
CMD ["python", "-u", "run_api.py"]

# =============================================================================
# Stage 4: Full ML Stage - Future upgrade path for complete ML capabilities
# =============================================================================
FROM builder as fullml

LABEL stage=fullml
LABEL purpose="Complete ML stack with torch, tensorflow, transformers"
LABEL cost_optimization="compute_intensive"
LABEL architecture="full_ml_models"

# Copy and install full ML dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application source code
COPY src/ ./src/
COPY run_api.py ./

# Set ownership and switch to non-root user
RUN chown -R apiuser:apiuser /app
USER apiuser

# Expose port
EXPOSE 8000

# Health check with longer start period for ML model loading
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Command with memory optimization for ML models
CMD ["python", "-u", "run_api.py"]

# =============================================================================
# Default Stage: Production (lightweight and cost-effective)
# =============================================================================
FROM production