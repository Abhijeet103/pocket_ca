from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag.models import utc_now_iso


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="User message to send to the chatbot.")
    user_id: str = Field(default="demo-user", min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Optional existing session id to continue a conversation.",
    )


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str
    responded_at: str = Field(default_factory=utc_now_iso)


class RebuildRequest(BaseModel):
    clear_graph: bool = True


class RebuildStatusResponse(BaseModel):
    status: str
    source: str | None = None
    clear_graph: bool | None = None
    job_id: str | None = None
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_error: str | None = None
    last_summary: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    scheduler_enabled: bool
    rebuild: RebuildStatusResponse
    checked_at: str = Field(default_factory=utc_now_iso)
