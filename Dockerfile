FROM python:3.12-slim

# Build-time env vars
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=wsgi.py \
    FLASK_ENV=production \
    PYTHONPATH=/app \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Install system dependencies first (better caching)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying code (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download de_core_news_sm && \
    python -m spacy download en_core_web_sm

# Copy application files
COPY . .

# Security hardening
RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    find /app -type d -exec chmod 755 {} \; && \
    find /app -type f -exec chmod 644 {} \; && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directory for gunicorn logs
RUN mkdir -p /var/log/gunicorn && \
    chown appuser:appuser /var/log/gunicorn

USER appuser

# Gunicorn config with improved settings
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--preload", \
     "wsgi:application"]
