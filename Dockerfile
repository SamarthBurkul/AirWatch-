FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ml_models

EXPOSE 5000

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5000", "--workers", "1"]
