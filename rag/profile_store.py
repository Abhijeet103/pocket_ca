from __future__ import annotations

from sqlalchemy import select

from rag.db import UserProfileRow, _upsert_user_profile_row, init_database, session_scope, user_profile_from_row
from rag.models import UserTaxProfile

# user repo layer


class UserProfileStore:
    def __init__(self) -> None:
        init_database()

    def list_profiles(self) -> list[UserTaxProfile]:
        with session_scope() as session:
            rows = session.scalars(
                select(UserProfileRow).order_by(UserProfileRow.user_id)
            ).all()
        return [user_profile_from_row(row) for row in rows]

    def get(self, user_id: str) -> UserTaxProfile | None:
        with session_scope() as session:
            row = session.get(UserProfileRow, user_id)
            return user_profile_from_row(row) if row else None

    def save(self, profile: UserTaxProfile) -> UserTaxProfile:
        profile.touch()
        with session_scope() as session:
            row = _upsert_user_profile_row(session, profile)
            return user_profile_from_row(row)

    def upsert(self, user_id: str, **updates) -> UserTaxProfile:
        existing = self.get(user_id) or UserTaxProfile(user_id=user_id)
        merged = existing.model_copy(update=updates)
        return self.save(merged)
