from __future__ import annotations

import re
from typing import Sequence

from llama_index.core.schema import NodeWithScore, TextNode

from rag.models import Citation


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_source_nodes(source_nodes: Sequence[NodeWithScore]) -> list[NodeWithScore]:
    deduped_nodes: list[NodeWithScore] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for node_with_score in source_nodes:
        metadata = node_with_score.node.metadata or {}
        key = (
            metadata.get("chunk_id"),
            metadata.get("source_file"),
            metadata.get("page_number"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_nodes.append(node_with_score)

    return deduped_nodes


def build_citations(source_nodes: Sequence[NodeWithScore]) -> list[Citation]:
    citations: list[Citation] = []
    for index, node_with_score in enumerate(dedupe_source_nodes(source_nodes), start=1):
        metadata = node_with_score.node.metadata or {}
        citations.append(
            Citation(
                label=f"[S{index}]",
                source_file=metadata.get("source_file", "Unknown"),
                page_number=metadata.get("page_number"),
                section_title=metadata.get("section_title"),
                statute_reference=metadata.get("statute_reference"),
                chunk_id=metadata.get("chunk_id"),
                score=node_with_score.score,
                excerpt=_compact_whitespace(node_with_score.node.text)[:320],
            )
        )
    return citations


def prepare_citation_context(
    source_nodes: Sequence[NodeWithScore],
) -> tuple[list[NodeWithScore], list[Citation]]:
    deduped_nodes = dedupe_source_nodes(source_nodes)
    citations = build_citations(deduped_nodes)
    citable_nodes: list[NodeWithScore] = []

    for citation, node_with_score in zip(citations, deduped_nodes):
        section_title = citation.section_title or "Unknown"
        statute_reference = citation.statute_reference or "Unknown"
        citable_text = (
            f"{citation.label}\n"
            f"Source: {citation.source_file}\n"
            f"Page: {citation.page_number}\n"
            f"Section: {section_title}\n"
            f"Reference: {statute_reference}\n"
            f"Excerpt:\n{node_with_score.node.text}"
        )
        citable_nodes.append(
            NodeWithScore(
                node=TextNode(
                    text=citable_text,
                    id_=node_with_score.node.node_id,
                    metadata=node_with_score.node.metadata,
                ),
                score=node_with_score.score,
            )
        )

    return citable_nodes, citations


def format_citations(citations: Sequence[Citation]) -> str:
    if not citations:
        return ""

    lines = ["Sources:"]
    for citation in citations:
        lines.append(
            f"{citation.label} {citation.source_file} | "
            f"page {citation.page_number} | "
            f"{citation.section_title or 'Unknown'}"
        )
    return "\n".join(lines)
