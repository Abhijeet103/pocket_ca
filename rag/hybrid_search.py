from __future__ import annotations

from rag.retriever import get_hybrid_retriever


def hybrid_search(query: str):
    return get_hybrid_retriever().retrieve(query)
