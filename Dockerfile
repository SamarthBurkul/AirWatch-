FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Only minimal OS packages
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install CPU-only Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Ensure model folder
RUN mkdir -p ml_models

# Download model from GitHub Release
ARG AQI_MODEL_URL
RUN if [ -n "$AQI_MODEL_URL" ]; then curl -L "$AQI_MODEL_URL" -o ml_models/random_forest_model.pkl; fi

EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000", "--workers", "1"]
