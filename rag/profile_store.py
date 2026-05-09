from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from rag.config import USER_PROFILE_STORE_PATH
from rag.models import UserTaxProfile
from rag.settings import ensure_storage_dirs


class UserProfileStore:
    def __init__(self, store_path: Path = USER_PROFILE_STORE_PATH) -> None:
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
            raise ValueError("User profile store is corrupted: expected a JSON object.")
        return parsed

    def _write_all(self, payload: dict[str, dict]) -> None:
        self._store_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def list_profiles(self) -> list[UserTaxProfile]:
        with self._lock:
            return [
                UserTaxProfile(**profile_data)
                for profile_data in self._read_all().values()
            ]

    def get(self, user_id: str) -> UserTaxProfile | None:
        with self._lock:
            payload = self._read_all()
            profile_data = payload.get(user_id)
            return UserTaxProfile(**profile_data) if profile_data else None

    def save(self, profile: UserTaxProfile) -> UserTaxProfile:
        with self._lock:
            payload = self._read_all()
            profile.touch()
            payload[profile.user_id] = profile.model_dump()
            self._write_all(payload)
            return profile

    def upsert(self, user_id: str, **updates) -> UserTaxProfile:
        existing = self.get(user_id) or UserTaxProfile(user_id=user_id)
        merged = existing.model_copy(update=updates)
        return self.save(merged)

