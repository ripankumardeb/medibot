from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from pinecone import Pinecone

from src.helper import download_embeddings, ensure_pinecone_index, get_llm
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
import logging


# ---------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Authenticate Hugging Face Hub downloads. Anonymous requests to the Hub are
# rate-limited, which is a common source of intermittent failures the first
# time the embedding model has to be downloaded/re-downloaded.
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
else:
    logging.getLogger(__name__).warning(
        "HF_TOKEN not set in .env - Hugging Face Hub downloads will be "
        "unauthenticated and may be rate-limited."
    )


# ---------------------------------------------------------
# RAG Setup (Modern LCEL — Python 3.14 compatible)
# ---------------------------------------------------------

MODEL_NAME = "llama3.2:3b"


logger.info("Loading embeddings...")
embeddings = download_embeddings()
embedding_dimension = embeddings.get_embedding_dimension()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", f"medical-chatbot-{embedding_dimension}")

logger.info("Connecting to Pinecone index: %s", INDEX_NAME)
pc = Pinecone(api_key=PINECONE_API_KEY)
ensure_pinecone_index(pc, INDEX_NAME, embedding_dimension)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


logger.info("Loading HF model...")
chat_model = get_llm()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL chain: retrieve → format → prompt → llm → parse
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("chatbot.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "running",
        "service": "medical-chatbot",
        "model": MODEL_NAME,
        "index": INDEX_NAME
    })


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()

        if not msg:
            return jsonify({
                "success": False,
                "answer": "Please type a message."
            }), 400

        if len(msg) > 1000:
            return jsonify({
                "success": False,
                "answer": "Please keep your question under 1000 characters."
            }), 400

        logger.info("User Message: %s", msg)

        answer = rag_chain.invoke(msg)

        logger.info("Bot Response: %s", answer)

        return jsonify({
            "success": True,
            "answer": answer
        })

    except ConnectionError as e:
        # Raised by OllamaLocalLLM when the local Ollama server can't be
        # reached after retries. This is a very common intermittent failure:
        # Ollama isn't running, is still starting up, or is loading the model
        # into memory for the first time.
        logger.exception("Ollama connection error: %s", str(e))

        return jsonify({
            "success": False,
            "answer": (
                "The AI model server (Ollama) isn't reachable right now. "
                "Make sure it's running (`ollama serve`) and that the model "
                "has been pulled, then try again."
            )
        }), 503

    except Exception as e:
        logger.exception("Chat Error: %s", str(e))

        return jsonify({
            "success": False,
            "answer": "Something went wrong. Please try again."
        }), 500


# ---------------------------------------------------------
# Main Entry
# ---------------------------------------------------------

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        use_reloader=False
    )