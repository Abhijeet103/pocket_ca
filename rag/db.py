from __future__ import annotations

import json
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock
from typing import Iterator
from uuid import uuid4

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.types import JSON

from rag.config import (
    DATABASE_URL,
    LEGACY_CHAT_SESSION_STORE_PATH,
    LEGACY_CHUNK_CATALOG_PATH,
    LEGACY_INGESTION_MANIFEST_PATH,
    LEGACY_USER_PROFILE_STORE_PATH,
    STORAGE_DIR,
)
from rag.models import ChatSession, ChatTurn, ChunkCatalogRecord, UserTaxProfile

# tables

class Base(DeclarativeBase):
    pass


class UserProfileRow(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    profession_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    tax_regime: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    financial_year: Mapped[str] = mapped_column(String(32), nullable=False)
    assessment_year: Mapped[str] = mapped_column(String(32), nullable=False)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    residential_status: Mapped[str] = mapped_column(String(64), nullable=False)
    employer_type: Mapped[str] = mapped_column(String(64), nullable=False)

    salary_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pension_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    freelance_receipts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    freelance_expenses: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    business_receipts: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    business_expenses: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    interest_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    savings_interest_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fixed_deposit_interest_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rental_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    other_income: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    capital_gains_special_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    use_presumptive_profession: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    presumptive_profession_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    use_presumptive_business: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    presumptive_business_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.08)

    salary_standard_deduction_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    exempt_allowances_old_regime: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    house_property_interest_self_occupied: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    employer_nps_contribution: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80c_total: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80ccd1b: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80d_self_family: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80d_parents: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80e_interest: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80g_donations: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    section_80cch_contribution: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    parents_are_senior_citizens: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    known_facts: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    missing_fields: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    turns: Mapped[list["ChatTurnRow"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatTurnRow.position",
    )


class ChatTurnRow(Base):
    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    session: Mapped[ChatSessionRow] = relationship(back_populates="turns")


class ChunkCatalogRow(Base):
    __tablename__ = "chunk_catalog"

    chunk_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    page_number: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    section_title: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    statute_reference: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    document_year: Mapped[str | None] = mapped_column(String(32), nullable=True)


class IngestionRunRow(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid4().hex)
    backend: Mapped[str] = mapped_column(String(64), nullable=False)
    source_files: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    document_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    neo4j_database: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(64), nullable=False, default="database")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


_PROFILE_FIELDS = tuple(UserTaxProfile.model_fields.keys())
_DATABASE_INIT_LOCK = Lock()
_DATABASE_INITIALIZED = False


def database_backend_name() -> str:
    return make_url(DATABASE_URL).get_backend_name()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    connect_args: dict[str, object] = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(DATABASE_URL, connect_args=connect_args)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    global _DATABASE_INITIALIZED

    if _DATABASE_INITIALIZED:
        return

    with _DATABASE_INIT_LOCK:
        if _DATABASE_INITIALIZED:
            return

        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            Base.metadata.create_all(get_engine())
        except OperationalError as exc:
            if "already exists" not in str(exc).lower():
                raise
        with session_scope() as session:
            _migrate_legacy_json_data(session)
        _DATABASE_INITIALIZED = True


def _read_json_object(path) -> dict | None:
    if not path.exists():
        return None
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return None
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return parsed


def _migrate_legacy_json_data(session: Session) -> None:
    if session.scalar(select(UserProfileRow).limit(1)) is None:
        profiles_payload = _read_json_object(LEGACY_USER_PROFILE_STORE_PATH)
        for raw_profile in (profiles_payload or {}).values():
            profile = UserTaxProfile.model_validate(raw_profile)
            _upsert_user_profile_row(session, profile)

    if session.scalar(select(ChatSessionRow).limit(1)) is None:
        sessions_payload = _read_json_object(LEGACY_CHAT_SESSION_STORE_PATH)
        for raw_session in (sessions_payload or {}).values():
            chat_session = ChatSession.model_validate(raw_session)
            _upsert_chat_session_row(session, chat_session)

    if session.scalar(select(ChunkCatalogRow).limit(1)) is None and LEGACY_CHUNK_CATALOG_PATH.exists():
        for line in LEGACY_CHUNK_CATALOG_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = ChunkCatalogRecord.model_validate_json(line)
            session.add(_chunk_record_to_row(record))

    if session.scalar(select(IngestionRunRow).limit(1)) is None:
        manifest_payload = _read_json_object(LEGACY_INGESTION_MANIFEST_PATH)
        if manifest_payload:
            session.add(
                IngestionRunRow(
                    run_id=uuid4().hex,
                    backend=str(manifest_payload.get("backend", "graph_rag")),
                    source_files=[
                        str(path)
                        for path in manifest_payload.get("source_files", [])
                    ],
                    document_count=int(manifest_payload.get("document_count", 0)),
                    page_count=int(manifest_payload.get("page_count", 0)),
                    chunk_count=int(manifest_payload.get("chunk_count", 0)),
                    neo4j_database=str(manifest_payload.get("neo4j_database", "neo4j")),
                    storage_backend="database",
                    created_at=manifest_payload.get("created_at") or "legacy-import",
                )
            )


def _upsert_user_profile_row(session: Session, profile: UserTaxProfile) -> UserProfileRow:
    row = session.get(UserProfileRow, profile.user_id)
    if row is None:
        row = UserProfileRow(user_id=profile.user_id)
        session.add(row)

    for field_name, value in profile.model_dump(mode="json").items():
        setattr(row, field_name, value)

    session.flush()
    return row


def user_profile_from_row(row: UserProfileRow) -> UserTaxProfile:
    payload = {
        field_name: getattr(row, field_name)
        for field_name in _PROFILE_FIELDS
    }
    return UserTaxProfile.model_validate(payload)


def _upsert_chat_session_row(session: Session, chat_session: ChatSession) -> ChatSessionRow:
    row = session.get(ChatSessionRow, chat_session.session_id)
    if row is None:
        row = ChatSessionRow(session_id=chat_session.session_id)
        session.add(row)

    payload = chat_session.model_dump(mode="json", exclude={"turns"})
    for field_name, value in payload.items():
        setattr(row, field_name, value)

    row.turns.clear()
    for position, turn in enumerate(chat_session.turns):
        row.turns.append(
            ChatTurnRow(
                position=position,
                role=turn.role,
                content=turn.content,
                name=turn.name,
                created_at=turn.created_at,
            )
        )

    session.flush()
    return row


def chat_session_from_row(row: ChatSessionRow) -> ChatSession:
    turns = [
        ChatTurn(
            role=turn.role,
            content=turn.content,
            name=turn.name,
            created_at=turn.created_at,
        )
        for turn in sorted(row.turns, key=lambda item: item.position)
    ]
    return ChatSession(
        session_id=row.session_id,
        user_id=row.user_id,
        title=row.title,
        turns=turns,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _chunk_record_to_row(record: ChunkCatalogRecord) -> ChunkCatalogRow:
    metadata = record.metadata
    chunk_index = metadata.get("chunk_index")
    return ChunkCatalogRow(
        chunk_id=record.chunk_id,
        text=record.text,
        metadata_json=metadata,
        source_file=metadata.get("source_file"),
        source_path=metadata.get("source_path"),
        document_id=metadata.get("document_id"),
        page_number=str(metadata.get("page_number")) if metadata.get("page_number") is not None else None,
        section_title=metadata.get("section_title"),
        statute_reference=metadata.get("statute_reference"),
        chunk_index=int(chunk_index) if chunk_index is not None else None,
        document_type=metadata.get("document_type"),
        document_year=str(metadata.get("document_year")) if metadata.get("document_year") is not None else None,
    )


def chunk_record_from_row(row: ChunkCatalogRow) -> ChunkCatalogRecord:
    return ChunkCatalogRecord(
        chunk_id=row.chunk_id,
        text=row.text,
        metadata=dict(row.metadata_json or {}),
    )
