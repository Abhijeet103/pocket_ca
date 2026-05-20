from __future__ import annotations

from typing import Iterable
from uuid import uuid4

from sqlalchemy import delete, desc, select

from rag.config import NEO4J_DATABASE
from rag.db import (
    ChunkCatalogRow,
    IngestionRunRow,
    chunk_record_from_row,
    init_database,
    session_scope,
    _chunk_record_to_row,
)
from rag.models import ChunkCatalogRecord, utc_now_iso


class ChunkCatalogStore:
    def __init__(self) -> None:
        init_database()

    def replace_all(self, records: Iterable[ChunkCatalogRecord]) -> int:
        records_list = list(records)
        with session_scope() as session:
            session.execute(delete(ChunkCatalogRow))
            for record in records_list:
                session.add(_chunk_record_to_row(record))
        return len(records_list)

    def list_records(self, limit: int | None = None) -> list[ChunkCatalogRecord]:
        with session_scope() as session:
            statement = select(ChunkCatalogRow).order_by(ChunkCatalogRow.chunk_id)
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
        return [chunk_record_from_row(row) for row in rows]


class IngestionRunStore:
    def __init__(self) -> None:
        init_database()

    def record_run(
        self,
        *,
        backend: str,
        source_files: list[str],
        document_count: int,
        page_count: int,
        chunk_count: int,
        neo4j_database: str = NEO4J_DATABASE,
    ) -> dict[str, object]:
        created_at = utc_now_iso()
        row = IngestionRunRow(
            run_id=uuid4().hex,
            backend=backend,
            source_files=source_files,
            document_count=document_count,
            page_count=page_count,
            chunk_count=chunk_count,
            neo4j_database=neo4j_database,
            storage_backend="database",
            created_at=created_at,
        )
        with session_scope() as session:
            session.add(row)
        return self.latest_run() or {
            "run_id": row.run_id,
            "backend": backend,
            "source_files": source_files,
            "document_count": document_count,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "neo4j_database": neo4j_database,
            "storage_backend": "database",
            "created_at": created_at,
        }

    def latest_run(self) -> dict[str, object] | None:
        with session_scope() as session:
            row = session.scalar(
                select(IngestionRunRow)
                .order_by(desc(IngestionRunRow.created_at))
                .limit(1)
            )
        if row is None:
            return None
        return {
            "run_id": row.run_id,
            "backend": row.backend,
            "source_files": list(row.source_files or []),
            "document_count": row.document_count,
            "page_count": row.page_count,
            "chunk_count": row.chunk_count,
            "neo4j_database": row.neo4j_database,
            "storage_backend": row.storage_backend,
            "created_at": row.created_at,
        }
