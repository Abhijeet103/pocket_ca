from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader

from rag.config import (
    CHUNK_CATALOG_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    INDEX_STORAGE_DIR,
    INGESTION_MANIFEST_PATH,
    SUPPORTED_SOURCE_SUFFIXES,
)
from rag.metadata_extractor import enrich_metadata, infer_document_type
from rag.models import ChunkCatalogRecord
from rag.settings import configure_settings, ensure_storage_dirs


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


def persist_chunk_catalog(records: list[ChunkCatalogRecord]) -> None:
    with CHUNK_CATALOG_PATH.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(record.model_dump_json() + "\n")


def persist_ingestion_manifest(
    source_files: list[Path],
    document_count: int,
    node_count: int,
) -> None:
    manifest = {
        "source_files": [str(path.resolve()) for path in source_files],
        "document_count": document_count,
        "node_count": node_count,
        "index_storage_dir": str(INDEX_STORAGE_DIR),
        "chunk_catalog_path": str(CHUNK_CATALOG_PATH),
    }
    INGESTION_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def ingest_documents(source_files: list[Path] | None = None) -> dict[str, int | str]:
    configure_settings()
    ensure_storage_dirs()

    resolved_files = source_files or discover_source_files()
    if not resolved_files:
        raise FileNotFoundError(
            f"No supported source files were found under {DATA_DIR}."
        )

    documents = load_documents(resolved_files)
    nodes, catalog_records = build_nodes(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(INDEX_STORAGE_DIR))

    persist_chunk_catalog(catalog_records)
    persist_ingestion_manifest(resolved_files, len(documents), len(nodes))
    from rag.query_engine import reset_query_engine_cache
    from rag.retriever import reset_retriever_cache

    reset_retriever_cache()
    reset_query_engine_cache()

    return {
        "source_files": len(resolved_files),
        "documents": len(documents),
        "nodes": len(nodes),
        "index_path": str(INDEX_STORAGE_DIR),
    }


def main() -> None:
    summary = ingest_documents()
    print(
        "Ingestion completed "
        f"(files={summary['source_files']}, "
        f"documents={summary['documents']}, "
        f"nodes={summary['nodes']})."
    )


if __name__ == "__main__":
    main()
