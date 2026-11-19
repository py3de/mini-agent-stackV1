"""Knowledge-base tool: lets the agent answer customer questions from company documents (RAG)."""

from __future__ import annotations

from langchain_core.tools import tool

from agent_stack.rag import get_vector_store


@tool
def search_knowledge_base(query: str) -> str:
    """Search company documents (security & hosting, services, contracts & pricing).
    Use this when the meeting notes contain a customer question or concern (e.g. data residency,
    timelines, payment terms) so the follow-up email answers it with facts, not guesses."""
    try:
        hits = get_vector_store().similarity_search(query, k=3)
    except Exception as exc:  # noqa: BLE001 - e.g. missing embeddings key
        return f"Knowledge base unavailable: {exc}"
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{d.metadata['source']} > {d.metadata['section']}]\n{d.page_content}" for d in hits
    )


KNOWLEDGE_TOOLS = [search_knowledge_base]
