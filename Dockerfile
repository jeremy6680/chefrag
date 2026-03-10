# ==============================================================================
# ChefRAG — Dockerfile
# Base: Python 3.11 slim (smaller image, faster build)
# ==============================================================================

FROM python:3.11-slim

# --- System dependencies ---
# curl: healthcheck; build-essential: some pip packages need C compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Python dependencies ---
# Copy requirements first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application code ---
COPY app/ ./app/
COPY tests/ ./tests/
COPY conftest.py ./conftest.py

# --- Storage directories (will be overridden by volumes in production) ---
RUN mkdir -p /app/storage/chroma /app/storage/duckdb

# --- Streamlit configuration ---
# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck: verify the app is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# --- Entrypoint ---
CMD ["streamlit", "run", "app/main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]