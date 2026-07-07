#!/bin/sh
# This container talks to Ollama over the host network (see
# `--network host` in the CI/CD deploy step and OLLAMA_BASE_URL in app.py).
# Ollama itself is NOT installed in this image - it must already be running
# on the host the container is deployed to (`ollama serve` +
# `ollama pull <model>`), otherwise every chat request will fail.
#
# To avoid the container accepting traffic before Ollama is actually ready
# (a common cause of "works sometimes, fails right after a fresh deploy"),
# wait here until Ollama responds, then start the app.

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
MAX_WAIT_SECONDS=120
waited=0

echo "Waiting for Ollama at ${OLLAMA_URL} ..."
until curl -s -o /dev/null "${OLLAMA_URL}"; do
    waited=$((waited + 2))
    if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
        echo "WARNING: Ollama at ${OLLAMA_URL} did not respond after ${MAX_WAIT_SECONDS}s. Starting the app anyway - chat requests will fail until Ollama is reachable."
        break
    fi
    sleep 2
done

exec gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:8080 app:app
