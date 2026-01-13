FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System libs needed for some Python wheels / compiling C/Fortran extensions
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# install numpy & scipy first (ensures wheels)
RUN pip install numpy==1.24.3 scipy==1.10.1

# copy requirements
COPY requirements.txt .

# install rest
RUN pip install -r requirements.txt

# copy project
COPY . .

# ensure model folder
RUN mkdir -p ml_models

# download model from GitHub Release (uses AQI_MODEL_URL env on Render)
ARG AQI_MODEL_URL=https://github.com/SamarthBurkul/AirWatch-/releases/download/v1.1.0/random_forest_model.pkl
RUN if [ -n "$AQI_MODEL_URL" ]; then curl -L "$AQI_MODEL_URL" -o ml_models/random_forest_model.pkl || true; fi

EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000", "--timeout", "120"]
