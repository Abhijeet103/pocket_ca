from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any
from uuid import uuid4

from rag.config import KNOWLEDGE_REBUILD_CLEAR_GRAPH
from rag.ingest import ingest_documents
from rag.models import utc_now_iso


class KnowledgeRebuildManager:
    def __init__(self) -> None:
        self._state_lock = Lock()
        self._status: dict[str, Any] = {
            "status": "idle",
            "source": None,
            "clear_graph": None,
            "job_id": None,
            "last_started_at": None,
            "last_finished_at": None,
            "last_error": None,
            "last_summary": None,
        }

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._status)

    def start_background_rebuild(
        self,
        *,
        clear_graph: bool | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        resolved_clear_graph = (
            KNOWLEDGE_REBUILD_CLEAR_GRAPH
            if clear_graph is None
            else clear_graph
        )

        with self._state_lock:
            if self._status["status"] == "running":
                raise RuntimeError("A knowledge-graph rebuild is already running.")

            job_id = uuid4().hex
            self._status.update(
                {
                    "status": "running",
                    "source": source,
                    "clear_graph": resolved_clear_graph,
                    "job_id": job_id,
                    "last_started_at": utc_now_iso(),
                    "last_finished_at": None,
                    "last_error": None,
                }
            )

        loop = asyncio.get_running_loop()
        loop.create_task(
            self._run_rebuild(
                job_id=job_id,
                clear_graph=resolved_clear_graph,
                source=source,
            )
        )
        return self.get_status()

    async def _run_rebuild(
        self,
        *,
        job_id: str,
        clear_graph: bool,
        source: str,
    ) -> None:
        try:
            summary = await asyncio.to_thread(
                ingest_documents,
                clear_graph=clear_graph,
            )
        except Exception as exc:  # noqa: BLE001
            with self._state_lock:
                if self._status.get("job_id") == job_id:
                    self._status.update(
                        {
                            "status": "failed",
                            "source": source,
                            "clear_graph": clear_graph,
                            "last_finished_at": utc_now_iso(),
                            "last_error": str(exc),
                        }
                    )
            return

        with self._state_lock:
            if self._status.get("job_id") == job_id:
                self._status.update(
                    {
                        "status": "completed",
                        "source": source,
                        "clear_graph": clear_graph,
                        "last_finished_at": utc_now_iso(),
                        "last_error": None,
                        "last_summary": summary,
                    }
                )

    async def scheduled_rebuild(self) -> None:
        try:
            self.start_background_rebuild(source="scheduler")
        except RuntimeError:
            return
