from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.config import (
    GRAPH_FINAL_TOP_K,
    GRAPH_KEYWORD_EXPANSION_TOP_K,
    GRAPH_NEIGHBOR_EXPANSION_TOP_K,
    GRAPH_REFERENCE_TOP_K,
    GRAPH_SECTION_TOP_K,
    GRAPH_TEXT_TOP_K,
)
from rag.graph_store import TaxLawGraphStore
from rag.graph_utils import extract_keywords


def reciprocal_rank_fusion_records(
    result_sets: Iterable[tuple[list[dict[str, Any]], float]],
    top_k: int,
    rank_constant: int = 60,
) -> list[tuple[dict[str, Any], float]]:
    combined_scores: dict[str, float] = defaultdict(float)
    record_lookup: dict[str, dict[str, Any]] = {}

    for result_set, weight in result_sets:
        for rank, record in enumerate(result_set, start=1):
            chunk_id = record["chunk_id"]
            combined_scores[chunk_id] += weight * (1.0 / (rank_constant + rank))
            record_lookup[chunk_id] = record

    fused_results = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (record_lookup[chunk_id], score)
        for chunk_id, score in fused_results[:top_k]
    ]


def _record_to_node(record: dict[str, Any], score: float) -> NodeWithScore:
    metadata = {
        "source_file": record.get("source_file", "Unknown"),
        "source_path": record.get("source_path", "Unknown"),
        "document_id": record.get("document_id", "unknown"),
        "document_type": record.get("document_type", "tax_reference"),
        "document_year": record.get("document_year", "Unknown"),
        "section_title": record.get("section_title", "Unknown"),
        "statute_reference": record.get("statute_reference", "Unknown"),
        "page_number": str(record.get("page_number", "Unknown")),
        "total_pages": str(record.get("total_pages", "Unknown")),
        "chunk_index": int(record.get("chunk_index", 0)),
        "chunk_id": record["chunk_id"],
    }
    node = TextNode(
        text=record.get("text", ""),
        id_=record["chunk_id"],
        metadata=metadata,
    )
    return NodeWithScore(node=node, score=score)


class GraphTaxLawRetriever(BaseRetriever):
    def __init__(self, top_k: int = GRAPH_FINAL_TOP_K) -> None:
        super().__init__()
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str.strip()
        if not query:
            return []

        query_keywords = extract_keywords(query)

        with TaxLawGraphStore() as store:
            store.ensure_schema()
            text_hits = store.search_chunk_text(query, limit=GRAPH_TEXT_TOP_K)
            section_hits = store.search_sections(query, limit=GRAPH_SECTION_TOP_K)
            reference_hits = store.search_references(query, limit=GRAPH_REFERENCE_TOP_K)

            seed_chunk_ids = list(
                dict.fromkeys(
                    [
                        record["chunk_id"]
                        for record in (text_hits + section_hits + reference_hits)
                    ]
                )
            )

            keyword_related_ids = store.fetch_shared_keyword_chunk_ids(
                seed_chunk_ids,
                keywords=query_keywords,
                per_seed_limit=GRAPH_KEYWORD_EXPANSION_TOP_K,
            )
            neighbor_chunk_ids = store.fetch_neighbor_chunk_ids(
                seed_chunk_ids,
                per_seed_limit=GRAPH_NEIGHBOR_EXPANSION_TOP_K,
            )

            keyword_related_hits = store.fetch_chunks_by_ids(keyword_related_ids)
            neighbor_hits = store.fetch_chunks_by_ids(neighbor_chunk_ids)

        fused_records = reciprocal_rank_fusion_records(
            [
                (text_hits, 1.0),
                (section_hits, 0.9),
                (reference_hits, 0.9),
                (keyword_related_hits, 0.55),
                (neighbor_hits, 0.5),
            ],
            top_k=self._top_k,
        )

        return [_record_to_node(record, score) for record, score in fused_records]


@lru_cache(maxsize=1)
def get_graph_retriever() -> GraphTaxLawRetriever:
    return GraphTaxLawRetriever(top_k=GRAPH_FINAL_TOP_K)


def get_hybrid_retriever() -> GraphTaxLawRetriever:
    return get_graph_retriever()


def reset_retriever_cache() -> None:
    get_graph_retriever.cache_clear()
