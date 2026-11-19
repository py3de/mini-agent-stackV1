import re

from langchain_core.embeddings import Embeddings

from agent_stack.rag import build_vector_store, load_documents, set_vector_store
from agent_stack.tools.knowledge import search_knowledge_base

VOCAB = ["hosting", "eu", "frankfurt", "gdpr", "payment", "invoiced", "weeks", "proof", "concept", "contract", "signatory", "migration"]


class KeywordEmbeddings(Embeddings):
    """Tiny bag-of-words embedding: deterministic and offline, but still 'semantic' enough to test retrieval."""

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [float(words.count(v)) + 1e-6 for v in VOCAB]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_documents_are_split_per_section_with_sources():
    docs = load_documents()
    sections = {(d.metadata["source"], d.metadata["section"]) for d in docs}
    assert ("security-and-hosting.md", "Data residency") in sections
    assert ("contracts-and-pricing.md", "Payment terms") in sections
    assert all(d.page_content.strip() for d in docs)


def test_knowledge_tool_retrieves_the_right_section():
    set_vector_store(build_vector_store(KeywordEmbeddings()))
    try:
        hosting = search_knowledge_base.invoke({"query": "Is our data hosted in the EU? Frankfurt?"})
        payment = search_knowledge_base.invoke({"query": "When is the project invoiced, payment terms?"})
    finally:
        set_vector_store(None)

    assert hosting.startswith("[security-and-hosting.md > Data residency]")
    assert payment.startswith("[contracts-and-pricing.md > Payment terms]")
