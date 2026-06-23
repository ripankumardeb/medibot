from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from src.helper import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# if OPENAI_API_KEY:
#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ---------------------------------------------------------
# RAG Setup
# ---------------------------------------------------------

INDEX_NAME = "medical-chatbot"
MODEL_NAME = "llama3.2:3b"


logger.info("Loading embeddings...")
embeddings = download_embeddings()


logger.info("Connecting to Pinecone index: %s", INDEX_NAME)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


logger.info("Loading Ollama model: %s", MODEL_NAME)
chat_model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt
)


rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
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

        response = rag_chain.invoke({
            "input": msg
        })

        answer = response.get(
            "answer",
            "Sorry, I could not find an answer."
        )

        logger.info("Bot Response: %s", answer)

        return jsonify({
            "success": True,
            "answer": answer
        })

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