from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Iterable

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.config import (
    CHUNK_CATALOG_PATH,
    HYBRID_TOP_K,
    INDEX_STORAGE_DIR,
    KEYWORD_TOP_K,
    SEMANTIC_TOP_K,
)
from rag.models import ChunkCatalogRecord
from rag.settings import configure_settings


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def reciprocal_rank_fusion(
    result_sets: Iterable[list[NodeWithScore]],
    top_k: int,
    rank_constant: int = 60,
) -> list[NodeWithScore]:
    combined_scores: dict[str, float] = defaultdict(float)
    node_lookup: dict[str, NodeWithScore] = {}

    for result_set in result_sets:
        for rank, node_with_score in enumerate(result_set, start=1):
            node_id = node_with_score.node.node_id
            combined_scores[node_id] += 1.0 / (rank_constant + rank)
            node_lookup[node_id] = node_with_score

    fused_results = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        NodeWithScore(node=node_lookup[node_id].node, score=score)
        for node_id, score in fused_results[:top_k]
    ]


@lru_cache(maxsize=1)
def load_index():
    configure_settings()
    if not INDEX_STORAGE_DIR.exists():
        raise FileNotFoundError(
            f"Index storage not found at {INDEX_STORAGE_DIR}. Run `python -m rag.ingest` first."
        )

    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_STORAGE_DIR))
    return load_index_from_storage(storage_context)


@lru_cache(maxsize=1)
def load_chunk_catalog() -> list[ChunkCatalogRecord]:
    if not CHUNK_CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Chunk catalog not found at {CHUNK_CATALOG_PATH}. Run `python -m rag.ingest` first."
        )

    records: list[ChunkCatalogRecord] = []
    with CHUNK_CATALOG_PATH.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                records.append(ChunkCatalogRecord(**json.loads(line)))
    return records


@lru_cache(maxsize=1)
def get_semantic_retriever() -> VectorIndexRetriever:
    return VectorIndexRetriever(index=load_index(), similarity_top_k=SEMANTIC_TOP_K)


class KeywordChunkRetriever(BaseRetriever):
    def __init__(
        self,
        catalog_records: list[ChunkCatalogRecord],
        top_k: int = KEYWORD_TOP_K,
    ) -> None:
        super().__init__()
        self._catalog_records = catalog_records
        self._top_k = top_k
        self._token_cache = {
            record.chunk_id: Counter(tokenize(record.text))
            for record in catalog_records
        }
        self._idf = self._build_inverse_document_frequency()

    def _build_inverse_document_frequency(self) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for tokens in self._token_cache.values():
            document_frequency.update(tokens.keys())

        document_count = max(len(self._token_cache), 1)
        return {
            token: math.log((document_count + 1) / (count + 1)) + 1
            for token, count in document_frequency.items()
        }

    def _score_record(self, query: str, query_tokens: list[str], record: ChunkCatalogRecord) -> float:
        document_tokens = self._token_cache[record.chunk_id]
        if not document_tokens:
            return 0.0

        query_counter = Counter(query_tokens)
        overlap_score = 0.0
        for token, token_count in query_counter.items():
            if token in document_tokens:
                overlap_score += min(token_count, document_tokens[token]) * self._idf.get(
                    token, 1.0
                )

        if overlap_score == 0:
            return 0.0

        coverage = len(set(query_tokens) & set(document_tokens)) / max(
            len(set(query_tokens)), 1
        )
        phrase_bonus = 1.0 if query.lower() in record.text.lower() else 0.0
        return overlap_score * (1.0 + coverage) + phrase_bonus

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str.strip()
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored_nodes: list[NodeWithScore] = []
        for record in self._catalog_records:
            score = self._score_record(query, query_tokens, record)
            if score <= 0:
                continue

            node = TextNode(
                text=record.text,
                id_=record.chunk_id,
                metadata=record.metadata,
            )
            scored_nodes.append(NodeWithScore(node=node, score=score))

        scored_nodes.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored_nodes[: self._top_k]


class HybridTaxLawRetriever(BaseRetriever):
    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        top_k: int = HYBRID_TOP_K,
    ) -> None:
        super().__init__()
        self._semantic_retriever = semantic_retriever
        self._keyword_retriever = keyword_retriever
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        semantic_nodes = self._semantic_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        return reciprocal_rank_fusion(
            [semantic_nodes, keyword_nodes],
            top_k=self._top_k,
        )


@lru_cache(maxsize=1)
def get_keyword_retriever() -> KeywordChunkRetriever:
    return KeywordChunkRetriever(load_chunk_catalog(), top_k=KEYWORD_TOP_K)


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridTaxLawRetriever:
    return HybridTaxLawRetriever(
        semantic_retriever=get_semantic_retriever(),
        keyword_retriever=get_keyword_retriever(),
        top_k=HYBRID_TOP_K,
    )


def reset_retriever_cache() -> None:
    load_index.cache_clear()
    load_chunk_catalog.cache_clear()
    get_semantic_retriever.cache_clear()
    get_keyword_retriever.cache_clear()
    get_hybrid_retriever.cache_clear()
