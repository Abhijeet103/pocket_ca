from __future__ import annotations

from rag.retriever import get_graph_retriever


def hybrid_search(query: str):
    return get_graph_retriever().retrieve(query)
