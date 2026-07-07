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
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8080

# NOTE: This image does not include Ollama. It must be running separately
# and reachable at OLLAMA_BASE_URL (defaults to http://localhost:11434,
# which works when the container is run with --network host on a machine
# that already has `ollama serve` running with the model pulled).
CMD ["./entrypoint.sh"]