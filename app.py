from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv

from src.helper import download_embeddings, get_llm
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
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# if OPENAI_API_KEY:
#     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ---------------------------------------------------------
# RAG Setup (Modern LCEL — Python 3.14 compatible)
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