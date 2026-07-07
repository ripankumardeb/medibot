import logging
import os
import time
from typing import Any, List, Optional

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


class SentenceTransformerEmbeddings(Embeddings):
    """
    Wraps sentence-transformers so it can be used as a LangChain Embeddings object.

    NOTE: this downloads the model from the Hugging Face Hub the first time it
    runs. Anonymous (unauthenticated) requests to the Hub are rate-limited, and
    on a busy network that rate limit is a common cause of requests that
    "sometimes work, sometimes fail". Passing an HF token (see
    download_embeddings() below) avoids that.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                self.model = SentenceTransformer(model_name, token=hf_token)
                break
            except Exception as exc:  # noqa: BLE001 - we want to retry on any download hiccup
                last_error = exc
                logger.warning(
                    "Failed to load embedding model (attempt %s/3): %s",
                    attempt, exc
                )
                time.sleep(2 * attempt)
        else:
            raise RuntimeError(
                f"Could not load embedding model '{model_name}' after 3 attempts. "
                f"Last error: {last_error}"
            )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [embedding.tolist() for embedding in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def get_embedding_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


def download_embeddings():
    return SentenceTransformerEmbeddings()


def ensure_pinecone_index(pc: Pinecone, index_name: str, embedding_dimension: int, timeout_seconds: int = 300) -> None:
    if pc.has_index(index_name):
        index_details = pc.describe_index(index_name)
        existing_dimension = getattr(index_details, "dimension", None) or index_details.get("dimension")

        if existing_dimension == embedding_dimension:
            logger.info("Pinecone index '%s' already matches embedding dimension %s.", index_name, embedding_dimension)
            return

        logger.warning(
            "Pinecone index '%s' has dimension %s but embedding model expects %s. Recreating index.",
            index_name,
            existing_dimension,
            embedding_dimension,
        )
        pc.delete_index(index_name)

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not pc.has_index(index_name):
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"Pinecone index '{index_name}' did not finish deleting in time.")

    logger.info("Creating or ensuring Pinecone index '%s' with dimension %s.", index_name, embedding_dimension)
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        index_details = pc.describe_index(index_name)
        if index_details.status["ready"]:
            return
        time.sleep(2)

    raise TimeoutError(f"Pinecone index '{index_name}' did not become ready in time.")


class OllamaLocalLLM(LLM):
    """
    Thin LangChain LLM wrapper around a local Ollama server.

    Fixes vs. the original version:
    - The underlying OllamaLLM client is created ONCE (cached on the instance)
      instead of on every single request.
    - `keep_alive` tells Ollama to keep the model loaded in memory instead of
      unloading it after ~5 minutes of idle time (Ollama's default). Without
      this, the FIRST request after any idle period has to reload the whole
      model from disk, which can easily take longer than a typical request
      timeout and shows up as a random/intermittent failure even though
      nothing is actually "broken".
    - Connection errors (Ollama not running yet, or still loading the model
      on a cold start) are retried a couple of times with backoff instead of
      failing immediately.
    """

    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _get_client(self) -> OllamaLLM:
        client = getattr(self, "_client", None)
        if client is None:
            client = OllamaLLM(
                model=self.model_name,
                temperature=0.2,
                num_predict=512,
                top_k=50,
                top_p=0.9,
                base_url=self.base_url,
                keep_alive="30m",
                validate_model_on_init=False,
            )
            object.__setattr__(self, "_client", client)
        return client

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        client = self._get_client()

        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                return client.invoke(prompt, stop=stop)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Ollama call failed (attempt %s/3): %s", attempt, exc
                )
                time.sleep(3 * attempt)

        raise ConnectionError(
            "Could not get a response from the local Ollama server at "
            f"{self.base_url}. Make sure Ollama is installed and running "
            f"(`ollama serve`) and that the model '{self.model_name}' has "
            f"been pulled (`ollama pull {self.model_name}`). "
            f"Last error: {last_error}"
        )


def get_llm():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaLocalLLM(model_name=model_name, base_url=base_url)
