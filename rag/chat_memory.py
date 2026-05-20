from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select

from rag.config import DEFAULT_CHAT_HISTORY_TURNS
from rag.db import ChatSessionRow, _upsert_chat_session_row, chat_session_from_row, init_database, session_scope
from rag.models import ChatSession

# repository laters

class ChatSessionStore:
    def __init__(self) -> None:
        init_database()

    def create_session_id(self, user_id: str) -> str:
        return f"{user_id}-{uuid4().hex[:8]}"

    def get(self, session_id: str) -> ChatSession | None:
        with session_scope() as session:
            row = session.get(ChatSessionRow, session_id)
            return chat_session_from_row(row) if row else None

    def save(self, chat_session: ChatSession) -> ChatSession:
        with session_scope() as session:
            row = _upsert_chat_session_row(session, chat_session)
            return chat_session_from_row(row)

    def get_or_create(
        self,
        user_id: str,
        session_id: str | None = None,
        title: str | None = None,
    ) -> ChatSession:
        if session_id:
            existing = self.get(session_id)
            if existing:
                return existing

        resolved_session_id = session_id or self.create_session_id(user_id)
        chat_session = ChatSession(
            session_id=resolved_session_id,
            user_id=user_id,
            title=title or f"Tax chat for {user_id}",
        )
        return self.save(chat_session)

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        name: str | None = None,
    ) -> ChatSession:
        chat_session = self.get(session_id)
        if chat_session is None:
            raise KeyError(f"Session {session_id} was not found.")
        chat_session.append_turn(role=role, content=content, name=name)
        return self.save(chat_session)

    def recent_messages(
        self,
        session_id: str,
        max_turns: int = DEFAULT_CHAT_HISTORY_TURNS,
    ) -> list[dict[str, str]]:
        chat_session = self.get(session_id)
        if chat_session is None:
            return []

        messages: list[dict[str, str]] = []
        for turn in chat_session.recent_turns(max_turns):
            if turn.role not in {"user", "assistant"}:
                continue
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def list_sessions(self, user_id: str | None = None) -> list[ChatSession]:
        with session_scope() as session:
            statement = select(ChatSessionRow).order_by(ChatSessionRow.updated_at.desc())
            if user_id:
                statement = statement.where(ChatSessionRow.user_id == user_id)
            rows = session.scalars(statement).all()
            return [chat_session_from_row(row) for row in rows]
