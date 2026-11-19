"""Retrieval over a small markdown knowledge base (RAG).

- Documents live in knowledge/*.md and are split into one chunk per `##` section, so every chunk
  is a self-contained answer with a clear source ("security-and-hosting.md > Data residency").
- Embeddings are model-agnostic like the chat models: EMBEDDINGS_PROVIDER picks the provider.
- The vector store is LangChain's InMemoryVectorStore: zero setup, rebuilt on start. Swapping it
  for pgvector / Supabase later only changes `build_vector_store`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from agent_stack.config import settings

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", Path(__file__).resolve().parents[2] / "knowledge"))


@dataclass(frozen=True)
class EmbeddingSpec:
    langchain_provider: str
    default_model: str
    api_key_env: str | None


EMBEDDING_PROVIDERS: dict[str, EmbeddingSpec] = {
    "google": EmbeddingSpec("google_genai", "gemini-embedding-2", "GOOGLE_API_KEY"),
    "openai": EmbeddingSpec("openai", "text-embedding-3-small", "OPENAI_API_KEY"),
    "ollama": EmbeddingSpec("ollama", "nomic-embed-text", None),
}


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Embedding model for the chosen provider (Anthropic has no embeddings API, so it falls back to Google)."""
    provider = (provider or os.getenv("EMBEDDINGS_PROVIDER") or settings.llm_provider).lower()
    if provider == "anthropic":
        provider = "google"
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"No embeddings for '{provider}'. Choose one of: {', '.join(EMBEDDING_PROVIDERS)}")
    spec = EMBEDDING_PROVIDERS[provider]
    if spec.api_key_env and not os.getenv(spec.api_key_env):
        raise RuntimeError(f"Embeddings provider '{provider}' needs {spec.api_key_env} (or set EMBEDDINGS_PROVIDER).")
    model = os.getenv(f"EMBEDDINGS_MODEL_{provider.upper()}", spec.default_model)
    return init_embeddings(model, provider=spec.langchain_provider)


def load_documents(directory: Path = KNOWLEDGE_DIR) -> list[Document]:
    """Split every markdown file into one Document per `##` section."""
    docs: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for section in re.split(r"\n(?=## )", text):
            lines = section.strip().splitlines()
            if not lines or not lines[0].startswith("## "):
                continue  # skip the file title / intro
            heading = lines[0].removeprefix("## ").strip()
            body = "\n".join(lines[1:]).strip()
            docs.append(
                Document(
                    page_content=f"{heading}\n{body}",
                    metadata={"source": path.name, "section": heading},
                )
            )
    return docs


def build_vector_store(embeddings: Embeddings | None = None, directory: Path = KNOWLEDGE_DIR) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embeddings or get_embeddings())
    store.add_documents(load_documents(directory))
    return store


_store: InMemoryVectorStore | None = None


def get_vector_store() -> InMemoryVectorStore:
    """Shared store, built lazily on first use (so nothing is embedded unless the tool is called)."""
    global _store
    if _store is None:
        _store = build_vector_store()
    return _store


def set_vector_store(store: InMemoryVectorStore | None) -> None:
    """Swap the store (used by tests and evals)."""
    global _store
    _store = store
