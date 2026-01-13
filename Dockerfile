# Dockerfile — production-ready for your app
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

WORKDIR /app

# System libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential gcc g++ libatlas-base-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install numpy & scipy first (wheels)
RUN python -m pip install numpy==1.24.3 scipy==1.10.1

# Copy requirements (caching layer)
COPY requirements.txt .

# Install the rest
RUN python -m pip install -r requirements.txt

# Copy source
COPY . .

# Allow passing model URL at build-time; if provided, download model
ARG AQI_MODEL_URL=""
RUN mkdir -p ml_models \
 && if [ -n "$AQI_MODEL_URL" ]; then \
      echo "Downloading model from $AQI_MODEL_URL" && \
      curl -fSL "$AQI_MODEL_URL" -o ml_models/random_forest_model.pkl || (echo "Model download failed" && exit 1); \
    else \
      echo "No AQI_MODEL_URL provided — expecting ml_models/random_forest_model.pkl in repo"; \
    fi

EXPOSE 5000

# Use wsgi:app if your entrypoint module is wsgi.py with `app = create_app()` or replace as needed
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000", "--workers", "3", "--timeout", "120"]
