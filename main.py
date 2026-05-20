from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.api_models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RebuildRequest,
    RebuildStatusResponse,
)
from rag.chatbot import TaxChatbot
from rag.config import (
    FRONTEND_DIR,
    KNOWLEDGE_REBUILD_CRON,
    KNOWLEDGE_REBUILD_ENABLED,
    KNOWLEDGE_REBUILD_TIMEZONE,
)
from rag.rebuild_manager import KnowledgeRebuildManager
from rag.settings import ensure_storage_dirs


def _build_scheduler(manager: KnowledgeRebuildManager) -> AsyncIOScheduler:
    timezone = ZoneInfo(KNOWLEDGE_REBUILD_TIMEZONE)
    trigger = CronTrigger.from_crontab(
        KNOWLEDGE_REBUILD_CRON,
        timezone=timezone,
    )
    scheduler = AsyncIOScheduler(timezone=timezone)
    scheduler.add_job(
        manager.scheduled_rebuild,
        trigger=trigger,
        id="knowledge_graph_rebuild",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_storage_dirs()
    rebuild_manager = KnowledgeRebuildManager()
    app.state.rebuild_manager = rebuild_manager
    app.state.scheduler = None

    if KNOWLEDGE_REBUILD_ENABLED:
        scheduler = _build_scheduler(rebuild_manager)
        scheduler.start()
        app.state.scheduler = scheduler

    try:
        yield
    finally:
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is not None:
            scheduler.shutdown(wait=False)


app = FastAPI(
    title="PocketCA Graph RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _frontend_index() -> Path:
    return FRONTEND_DIR / "index.html"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    index_path = _frontend_index()
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.get("/chat", include_in_schema=False)
async def chat_ui() -> FileResponse:
    return await index()


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    rebuild_manager: KnowledgeRebuildManager = app.state.rebuild_manager
    return HealthResponse(
        status="ok",
        scheduler_enabled=KNOWLEDGE_REBUILD_ENABLED,
        rebuild=RebuildStatusResponse.model_validate(rebuild_manager.get_status()),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        chatbot = TaxChatbot(
            user_id=payload.user_id,
            session_id=payload.session_id,
        )
        response_text = await asyncio.to_thread(chatbot.chat, payload.message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        user_id=payload.user_id,
        session_id=chatbot.session_id,
        response=response_text,
    )


@app.get("/api/rebuild/status", response_model=RebuildStatusResponse)
async def rebuild_status() -> RebuildStatusResponse:
    rebuild_manager: KnowledgeRebuildManager = app.state.rebuild_manager
    return RebuildStatusResponse.model_validate(rebuild_manager.get_status())


@app.post("/api/rebuild", response_model=RebuildStatusResponse, status_code=202)
async def trigger_rebuild(payload: RebuildRequest) -> RebuildStatusResponse:
    rebuild_manager: KnowledgeRebuildManager = app.state.rebuild_manager
    try:
        status = rebuild_manager.start_background_rebuild(
            clear_graph=payload.clear_graph,
            source="manual_api",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RebuildStatusResponse.model_validate(status)
