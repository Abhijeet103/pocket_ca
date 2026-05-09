from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "rag"
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_STORAGE_DIR = STORAGE_DIR / "index"
CHUNK_CATALOG_PATH = STORAGE_DIR / "chunk_catalog.jsonl"
INGESTION_MANIFEST_PATH = STORAGE_DIR / "ingestion_manifest.json"
USER_PROFILE_STORE_PATH = STORAGE_DIR / "user_profiles.json"
CHAT_SESSION_STORE_PATH = STORAGE_DIR / "chat_sessions.json"
SYSTEM_PROMPT_PATH = RAG_DIR / "system_prompt.txt"

SUPPORTED_SOURCE_SUFFIXES = {".pdf"}

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
SEMANTIC_TOP_K = int(os.getenv("RAG_SEMANTIC_TOP_K", "6"))
KEYWORD_TOP_K = int(os.getenv("RAG_KEYWORD_TOP_K", "6"))
HYBRID_TOP_K = int(os.getenv("RAG_HYBRID_TOP_K", "8"))

DEFAULT_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_LLM_MODEL)
DEFAULT_RESPONSE_MODE = os.getenv("RAG_RESPONSE_MODE", "compact")
DEFAULT_CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "12"))
DEFAULT_CHAT_TOOL_STEPS = int(os.getenv("CHAT_TOOL_STEPS", "6"))

DEFAULT_FINANCIAL_YEAR = os.getenv("DEFAULT_FINANCIAL_YEAR", "FY 2025-26")
DEFAULT_ASSESSMENT_YEAR = os.getenv("DEFAULT_ASSESSMENT_YEAR", "AY 2026-27")
DEFAULT_STANDARD_DEDUCTION = float(
    os.getenv("DEFAULT_STANDARD_DEDUCTION", "75000")
)
