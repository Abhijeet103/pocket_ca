from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from rag.config import (
    CHUNK_CATALOG_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INGESTION_MANIFEST_PATH,
    NEO4J_DATABASE,
    SUPPORTED_SOURCE_SUFFIXES,
)
from rag.graph_store import TaxLawGraphStore
from rag.graph_utils import extract_keywords, normalize_text
from rag.metadata_extractor import enrich_metadata, infer_document_type
from rag.models import ChunkCatalogRecord
from rag.settings import ensure_storage_dirs


def discover_source_files(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
    )


def load_documents(source_files: Iterable[Path]) -> list:
    reader = PyMuPDFReader()
    documents = []

    for path in source_files:
        extra_info = {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "document_type": infer_document_type(path),
        }
        loaded_docs = reader.load_data(file_path=str(path), extra_info=extra_info)

        for doc in loaded_docs:
            doc.metadata.update(extra_info)

        documents.extend(loaded_docs)

    return documents


def build_nodes(documents: list) -> tuple[list, list[ChunkCatalogRecord]]:
    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = parser.get_nodes_from_documents(documents)
    catalog_records: list[ChunkCatalogRecord] = []

    for idx, node in enumerate(nodes):
        metadata = enrich_metadata(node, idx)
        node.metadata = metadata
        node.id_ = metadata["chunk_id"]
        catalog_records.append(
            ChunkCatalogRecord(
                chunk_id=metadata["chunk_id"],
                text=node.text,
                metadata=metadata,
            )
        )

    return nodes, catalog_records


def _page_sort_key(page_number: str) -> tuple[int, str]:
    if str(page_number).isdigit():
        return (0, f"{int(page_number):08d}")
    return (1, str(page_number))


def build_graph_documents(records: list[ChunkCatalogRecord]) -> list[dict]:
    documents_map: dict[str, dict] = {}
    page_maps: dict[str, dict[str, dict]] = defaultdict(dict)

    sorted_records = sorted(
        records,
        key=lambda record: (
            str(record.metadata.get("document_id", "")),
            _page_sort_key(str(record.metadata.get("page_number", "Unknown"))),
            int(record.metadata.get("chunk_index", 0)),
        ),
    )

    for record in sorted_records:
        metadata = record.metadata
        document_id = str(metadata.get("document_id", "unknown"))
        page_number = str(metadata.get("page_number", "Unknown"))

        if document_id not in documents_map:
            documents_map[document_id] = {
                "document_id": document_id,
                "name": metadata.get("source_file", "Unknown"),
                "path": metadata.get("source_path", "Unknown"),
                "document_type": metadata.get("document_type", "tax_reference"),
                "document_year": metadata.get("document_year", "Unknown"),
                "pages": [],
            }

        page_map = page_maps[document_id]
        if page_number not in page_map:
            page_map[page_number] = {
                "page_id": f"{document_id}-p{page_number}",
                "page_number": page_number,
                "preview": normalize_text(record.text)[:240],
                "chunks": [],
            }

        page_map[page_number]["chunks"].append(
            {
                "chunk_id": record.chunk_id,
                "text": record.text,
                "metadata": metadata,
                "keywords": extract_keywords(record.text),
            }
        )

    graph_documents: list[dict] = []
    for document_id, document_payload in documents_map.items():
        pages = list(page_maps[document_id].values())
        pages.sort(key=lambda page: _page_sort_key(str(page["page_number"])))
        for page in pages:
            page["chunks"].sort(key=lambda chunk: int(chunk["metadata"].get("chunk_index", 0)))
        document_payload["pages"] = pages
        graph_documents.append(document_payload)

    return graph_documents


def persist_chunk_catalog(records: list[ChunkCatalogRecord]) -> None:
    with CHUNK_CATALOG_PATH.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(record.model_dump_json() + "\n")


def persist_ingestion_manifest(
    source_files: list[Path],
    document_count: int,
    page_count: int,
    chunk_count: int,
) -> None:
    manifest = {
        "backend": "graph_rag",
        "source_files": [str(path.resolve()) for path in source_files],
        "document_count": document_count,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "neo4j_database": NEO4J_DATABASE,
        "chunk_catalog_path": str(CHUNK_CATALOG_PATH),
    }
    INGESTION_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def ingest_documents(
    source_files: list[Path] | None = None,
    clear_graph: bool = False,
) -> dict[str, int | str]:
    ensure_storage_dirs()

    resolved_files = source_files or discover_source_files()
    if not resolved_files:
        raise FileNotFoundError(
            f"No supported source files were found under {DATA_DIR}."
        )

    documents = load_documents(resolved_files)
    nodes, catalog_records = build_nodes(documents)
    graph_documents = build_graph_documents(catalog_records)

    with TaxLawGraphStore() as store:
        if clear_graph:
            store.clear_graph()
        store.ensure_schema()
        store.ingest_documents(graph_documents)

    persist_chunk_catalog(catalog_records)
    persist_ingestion_manifest(
        resolved_files,
        len(graph_documents),
        len(documents),
        len(nodes),
    )
    from rag.query_engine import reset_query_engine_cache
    from rag.retriever import reset_retriever_cache

    reset_retriever_cache()
    reset_query_engine_cache()

    return {
        "backend": "graph_rag",
        "source_files": len(resolved_files),
        "documents": len(documents),
        "nodes": len(nodes),
        "graph_documents": len(graph_documents),
        "neo4j_database": NEO4J_DATABASE,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest tax-law source documents into Neo4j.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing graph data for this app's labels before ingesting.",
    )
    args = parser.parse_args()

    summary = ingest_documents(clear_graph=args.clear)
    print(
        "Graph ingestion completed "
        f"(files={summary['source_files']}, "
        f"documents={summary['documents']}, "
        f"chunks={summary['nodes']}, "
        f"database={summary['neo4j_database']})."
    )


if __name__ == "__main__":
    main()
