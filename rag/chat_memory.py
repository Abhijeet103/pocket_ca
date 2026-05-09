from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from rag.config import CHAT_SESSION_STORE_PATH, DEFAULT_CHAT_HISTORY_TURNS
from rag.models import ChatSession
from rag.settings import ensure_storage_dirs


class ChatSessionStore:
    def __init__(self, store_path: Path = CHAT_SESSION_STORE_PATH) -> None:
        ensure_storage_dirs()
        self._store_path = store_path
        self._lock = RLock()

    def _read_all(self) -> dict[str, dict]:
        if not self._store_path.exists():
            return {}

        raw_text = self._store_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return {}

        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Chat session store is corrupted: expected a JSON object.")
        return parsed

    def _write_all(self, payload: dict[str, dict]) -> None:
        self._store_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def create_session_id(self, user_id: str) -> str:
        return f"{user_id}-{uuid4().hex[:8]}"

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            payload = self._read_all()
            session_data = payload.get(session_id)
            return ChatSession(**session_data) if session_data else None

    def save(self, session: ChatSession) -> ChatSession:
        with self._lock:
            payload = self._read_all()
            payload[session.session_id] = session.model_dump(mode="json")
            self._write_all(payload)
            return session

    def get_or_create(
        self,
        user_id: str,
        session_id: str | None = None,
        title: str | None = None,
    ) -> ChatSession:
        with self._lock:
            if session_id:
                existing = self.get(session_id)
                if existing:
                    return existing

            resolved_session_id = session_id or self.create_session_id(user_id)
            session = ChatSession(
                session_id=resolved_session_id,
                user_id=user_id,
                title=title or f"Tax chat for {user_id}",
            )
            return self.save(session)

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        name: str | None = None,
    ) -> ChatSession:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} was not found.")
        session.append_turn(role=role, content=content, name=name)
        return self.save(session)

    def recent_messages(
        self,
        session_id: str,
        max_turns: int = DEFAULT_CHAT_HISTORY_TURNS,
    ) -> list[dict[str, str]]:
        session = self.get(session_id)
        if session is None:
            return []

        messages: list[dict[str, str]] = []
        for turn in session.recent_turns(max_turns):
            if turn.role not in {"user", "assistant"}:
                continue
            messages.append({"role": turn.role, "content": turn.content})
        return messages
