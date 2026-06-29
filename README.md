# Medibot – Medical Chatbot

A Flask-based medical question-answering web app that uses retrieval-augmented generation (RAG) with Pinecone, Sentence-Transformers, and Ollama.

## What this repo contains

- `app.py` – Flask application and API endpoints
- `src/helper.py` – embedding and LLM helper classes
- `src/prompt.py` – system prompt used by the RAG pipeline
- `templates/chatbot.html` – frontend UI
- `requirements.txt` – Python dependencies
- `Dockerfile` – container build instructions

## Clone the repository

```powershell
cd C:\Users\ripanj\Documents\internship
git clone https://github.com/<your-username>/medibot.git
cd medibot
```

> Replace `https://github.com/<your-username>/medibot.git` with your repository URL.

## Setup on Windows

1. Create and activate a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Create a `.env` file with your Pinecone API key:

```powershell
@"
PINECONE_API_KEY=your_pinecone_api_key_here
"@ | Out-File -Encoding utf8 .env
```

4. Install and run Ollama locally:

```powershell
# Install Ollama if not installed
# https://ollama.com/download

ollama pull llama3.2:3b
ollama serve
```

5. Ensure the Pinecone index exists and is named `medical-chatbot`.

## Run the app

```powershell
.\venv\Scripts\python.exe app.py
```

Open a browser and visit:

- `http://127.0.0.1:8080`
- or `http://<your-local-ip>:8080`

## Usage

- Type a medical question into the chat UI.
- The app uses the local embedding model to encode the query.
- It performs a similarity search against Pinecone.
- It sends the retrieved context to Ollama for answer generation.

## Configuration

Use `.env` to set values:

- `PINECONE_API_KEY` – required
- `PINECONE_ENVIRONMENT` – optional
- `PINECONE_INDEX_NAME` – optional, defaults to `medical-chatbot`

## Common troubleshooting

- `Command not found: ollama` → Install Ollama and add it to your PATH.
- `Unable to connect to Ollama` → Run `ollama serve` and confirm `llama3.2:3b` is available.
- `Pinecone authentication failed` → Check `PINECONE_API_KEY` and verify the index exists.
- `App 500 error on /get` → Check the server logs for model or network failures.

## Notes

- This project expects a local Ollama server on `http://localhost:11434`.
- The app uses a `sentence-transformers` embedding model locally.
- If you prefer Docker, build the image with `docker build -t medibot .` and run it with Docker.

## License

Apache License 2.0
