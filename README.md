# Medibot – Medical Chatbot with Retrieval‑Augmented Generation

## Overview
Medibot is a Flask‑based web application that answers medical queries using a Retrieval‑Augmented Generation (RAG) pipeline. It leverages:
- **Pinecone** for vector storage (index `medical-chatbot`).
- **Sentence‑Transformers** (`all‑MiniLM‑L6‑v2`) for embeddings.
- **Ollama** (`llama3.2:3b`) as the LLM backend.
- **LangChain LCEL** to glue the components together.

The app runs locally on port **8080** and can be accessed from any browser on the same machine or LAN.

---
## Prerequisites
- Windows 10/11 with **Python 3.14** installed.
- [**Ollama**](https://ollama.com/) installed and the model `llama3.2:3b` pulled (`ollama pull llama3.2:3b`).
- A **Pinecone** account; you need an API key and an index named `medical-chatbot`.
- PowerShell (or Command Prompt) for running the commands.

---
## Quick Start
```powershell
# 1️⃣  Navigate to the project folder
cd "C:\Users\ripan\Documents\internship\medibot"

<<<<<<< HEAD
# 2️⃣  (Optional) Create a virtual environment
python -m venv venv

# 3️⃣  Activate the virtual environment
.\venv\Scripts\Activate.ps1   # PowerShell
#   or
.\venv\Scripts\activate.bat   # Cmd.exe

# 4️⃣  Install dependencies
.\venv\Scripts\python -m pip install -U -r requirements.txt

# 5️⃣  Store your Pinecone API key (replace with your own key if different)
@"
PINECONE_API_KEY=pcsk_ukgxD_6n5KrAespJcY7CBqLhkFpSNepDSoTFJH1bX7QsN8Fj5PqJEFXsgZeuCVF464aW7
"@ | Out-File -Encoding utf8 .env

# 6️⃣  Run the Flask app
.\venv\Scripts\python.exe app.py
```
When the server starts you will see:
```
* Running on http://127.0.0.1:8080
* Running on http://<LAN‑IP>:8080
```
Open one of those URLs in a browser to start chatting with Medibot.

---
## Configuration
- **`.env`** – environment variables used by the app:
  - `PINECONE_API_KEY` – your Pinecone API key (required).
  - `PINECONE_ENVIRONMENT` – optional, defaults to the environment set in Pinecone.
  - `PINECONE_INDEX_NAME` – defaults to `medical-chatbot`.
- **Ollama** – ensure the model `llama3.2:3b` is available (`ollama list`). The app talks to Ollama on `http://localhost:11434`.

---
## How It Works
1. **Embedding** – The user's question is encoded with the Sentence‑Transformer model.
2. **Vector Search** – The embedding queries Pinecone for the most relevant documents.
3. **Prompt Construction** – Retrieved docs are inserted into a system prompt (`src/prompt.py`).
4. **LLM Generation** – The prompt is sent to Ollama’s LLM which returns an answer.
5. **Flask API** – `POST /chat` accepts JSON `{ "question": "..." }` and replies `{ "answer": "..." }`.

---
## Development
- Run in debug mode (auto‑reload):
```powershell
$env:FLASK_ENV="development"
.\venv\Scripts\python.exe app.py
```
- Tests (if any) can be executed with:
```powershell
.\venv\Scripts\python -m pytest
```
- Adjust logging level in `app.py` for more/less verbosity.

---
## Troubleshooting
| Issue | Fix |
|-------|-----|
| Import errors | Re‑install dependencies: `pip install -U -r requirements.txt` |
| Ollama cannot connect | Start Ollama (`ollama serve`) and ensure the model is pulled |
| Pinecone authentication fails | Verify `PINECONE_API_KEY` and that the index `medical-chatbot` exists |
| Flask not reachable on LAN | Open port 8080 in Windows Firewall |

---
## License
This project is provided under the **MIT License**. See `LICENSE` for details.

---
## Acknowledgements
- **LangChain** – RAG framework
- **Pinecone** – Vector database
- **Ollama** – Local LLM serving
- **Sentence‑Transformers** – Embedding model

Enjoy building with Medibot! 🎉
=======
#pip install -r requirements.txt

#794285266297.dkr.ecr.ap-southeast-2.amazonaws.com/medibot
>>>>>>> 8ca0750a32f2a8d0224a3cc61a0617668648699e
