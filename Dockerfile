FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .
COPY src ./src

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates ./templates
COPY static ./static

EXPOSE 8080

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:8080", "app:app"]