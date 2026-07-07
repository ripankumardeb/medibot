# Medibot – Medical Chatbot

A Flask-based medical question-answering web app using Retrieval-Augmented
Generation (RAG):

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- **Vector store:** [Pinecone](https://www.pinecone.io/)
- **LLM:** [Ollama](https://ollama.com/) running locally (`llama3.2:3b` by default)
- **Server:** Flask + Gunicorn

> The chatbot only answers from the content of `data/Medical_book.pdf` (or
> whatever PDFs you put in `data/`). It's a reference tool, not a substitute
> for professional medical advice.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Flask app and API routes (`/`, `/get`, `/health`) |
| `store_index.py` | One-off script: loads PDFs, chunks them, embeds them, and upserts into Pinecone |
| `src/helper.py` | Embedding model + Ollama LLM wrapper |
| `src/prompt.py` | System prompt for the RAG chain |
| `templates/chatbot.html` | Frontend UI |
| `static/style.css` | Frontend styling |
| `data/` | Source PDF(s) to be indexed |
| `Dockerfile` / `entrypoint.sh` | Container build + startup |
| `.github/workflows/cicd.yaml` | Build → push to ECR → deploy to EC2 |

## Prerequisites

- Python 3.10–3.12
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Hugging Face](https://huggingface.co/) account and access token
  (used to download the embedding model without hitting anonymous rate limits)
- [Ollama](https://ollama.com/download) installed locally, with the model pulled:

  ```bash
  ollama pull llama3.2:3b
  ```

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/medibot.git
cd medibot
```

## 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs, among other things, `sentence-transformers` (and its `torch`
dependency) — the first install can take a few minutes and a few hundred MB
of disk space.

## 4. Configure environment variables

Copy the example file and fill in your real values:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

`.env`:

```dotenv
PINECONE_API_KEY=your_pinecone_api_key_here
HF_TOKEN=your_huggingface_token_here

# Optional overrides — defaults shown
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

`.env` is already in `.gitignore` — never commit real keys. If a key is ever
pasted somewhere outside your own machine (a chat, a shared doc, a public
repo), rotate it from the Pinecone / Hugging Face dashboard.

## 5. Start Ollama

In a separate terminal, leave this running:

```bash
ollama serve
```

(On macOS/Windows with the Ollama desktop app, this runs automatically in the
background — you can skip this step.)

## 6. Build the Pinecone index

Run this once, or again whenever you change the PDFs in `data/`:

```bash
python store_index.py
```

This creates the `medical-chatbot` Pinecone index (if it doesn't exist yet),
waits for it to become ready, chunks the PDF(s), embeds each chunk, and
upserts them into the index.

## 7. Run the app

```bash
python app.py
```

Then open:

- http://127.0.0.1:8080

## How a request flows

1. Your message is sent to `/get`.
2. It's embedded with the local `sentence-transformers` model.
3. Pinecone returns the top 3 most similar chunks from the medical PDF.
4. Those chunks + your question are sent to the local Ollama model as context.
5. Ollama's answer is returned as JSON and rendered in the chat UI.

## Configuration reference (`.env`)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `PINECONE_API_KEY` | Yes | — | App won't start without it |
| `HF_TOKEN` | Recommended | — | Avoids Hugging Face Hub rate limits on model download |
| `OLLAMA_MODEL` | No | `llama3.2:3b` | Must already be pulled via `ollama pull` |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Change if Ollama runs elsewhere (e.g. a different host/port) |

## Running with Docker

The image does **not** bundle Ollama — Ollama must be installed and running
on the host machine the container runs on.

```bash
docker build -t medibot .
docker run -d --name medibot \
  --network host \
  -e PINECONE_API_KEY=your_pinecone_api_key_here \
  -e HF_TOKEN=your_huggingface_token_here \
  medibot
```

`--network host` lets the container reach Ollama at `http://localhost:11434`
on the host. The container's `entrypoint.sh` waits (up to 2 minutes) for
Ollama to respond before starting Gunicorn, to avoid failing requests during
the first few seconds after a fresh deploy.

## Deployment (EC2 via GitHub Actions)

`.github/workflows/cicd.yaml` builds the Docker image, pushes it to ECR, then
runs it on a self-hosted EC2 runner with `--network host`. For this to work,
the EC2 instance itself must have Ollama installed, running, and the model
pulled — the same as any other host. Required GitHub secrets:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (or `AWS_REGION`)
- `PINECONE_API_KEY`
- `HF_TOKEN`

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Dependencies installed before it was added to `requirements.txt`, or an old/partial install | `pip install -r requirements.txt` again in a clean venv |
| `Command not found: ollama` | Ollama not installed | Install from https://ollama.com/download |
| Works, then randomly 503s / "AI model server isn't reachable" | Ollama not running, still starting up, or the model isn't pulled | Run `ollama serve` and `ollama pull llama3.2:3b`; the app retries a few times automatically before giving up |
| First request after the app has been idle for a while is slow/times out | Ollama unloaded the model from memory (default idle timeout) | Already mitigated with `keep_alive="30m"` in `src/helper.py`; increase it if you have longer idle gaps |
| `Pinecone authentication failed` | Wrong or missing `PINECONE_API_KEY` | Check `.env` and that the index name matches `medical-chatbot` |
| Errors right after running `store_index.py` for the first time | Querying a brand-new Pinecone index before it's fully ready | Already mitigated — `store_index.py` now waits for the index to report "ready" |
| Slow/failed model download on first run | Hugging Face Hub rate-limiting anonymous requests | Set `HF_TOKEN` in `.env` |
| `App 500 error on /get` | Check the server logs — they now print the actual exception type and message |

## License

See `LICENSE`.
