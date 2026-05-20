from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

RAG_DIR = BASE_DIR / "rag"
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
FRONTEND_DIR = BASE_DIR / "frontend"
DATABASE_PATH = STORAGE_DIR / "pocketca.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
LEGACY_CHUNK_CATALOG_PATH = STORAGE_DIR / "chunk_catalog.jsonl"
LEGACY_INGESTION_MANIFEST_PATH = STORAGE_DIR / "ingestion_manifest.json"
LEGACY_USER_PROFILE_STORE_PATH = STORAGE_DIR / "user_profiles.json"
LEGACY_CHAT_SESSION_STORE_PATH = STORAGE_DIR / "chat_sessions.json"
SYSTEM_PROMPT_PATH = RAG_DIR / "system_prompt.txt"

SUPPORTED_SOURCE_SUFFIXES = {".pdf"}

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MAX_KEYWORDS_PER_CHUNK = int(os.getenv("RAG_MAX_KEYWORDS_PER_CHUNK", "10"))

GRAPH_TEXT_TOP_K = int(os.getenv("RAG_GRAPH_TEXT_TOP_K", "6"))
GRAPH_SECTION_TOP_K = int(os.getenv("RAG_GRAPH_SECTION_TOP_K", "4"))
GRAPH_REFERENCE_TOP_K = int(os.getenv("RAG_GRAPH_REFERENCE_TOP_K", "4"))
GRAPH_NEIGHBOR_EXPANSION_TOP_K = int(
    os.getenv("RAG_GRAPH_NEIGHBOR_EXPANSION_TOP_K", "1")
)
GRAPH_KEYWORD_EXPANSION_TOP_K = int(
    os.getenv("RAG_GRAPH_KEYWORD_EXPANSION_TOP_K", "1")
)
GRAPH_FINAL_TOP_K = int(os.getenv("RAG_GRAPH_FINAL_TOP_K", "8"))

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DEFAULT_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_LLM_MODEL)
DEFAULT_RESPONSE_MODE = os.getenv("RAG_RESPONSE_MODE", "compact")
DEFAULT_CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "12"))
DEFAULT_CHAT_TOOL_STEPS = int(os.getenv("CHAT_TOOL_STEPS", "6"))

KNOWLEDGE_REBUILD_ENABLED = os.getenv(
    "KNOWLEDGE_REBUILD_ENABLED",
    "true",
).lower() in {"1", "true", "yes", "on"}
KNOWLEDGE_REBUILD_CRON = os.getenv("KNOWLEDGE_REBUILD_CRON", "0 3 * * *")
KNOWLEDGE_REBUILD_TIMEZONE = os.getenv(
    "KNOWLEDGE_REBUILD_TIMEZONE",
    "Asia/Kolkata",
)
KNOWLEDGE_REBUILD_CLEAR_GRAPH = os.getenv(
    "KNOWLEDGE_REBUILD_CLEAR_GRAPH",
    "true",
).lower() in {"1", "true", "yes", "on"}

DEFAULT_FINANCIAL_YEAR = os.getenv("DEFAULT_FINANCIAL_YEAR", "FY 2025-26")
DEFAULT_ASSESSMENT_YEAR = os.getenv("DEFAULT_ASSESSMENT_YEAR", "AY 2026-27")
DEFAULT_STANDARD_DEDUCTION = float(
    os.getenv("DEFAULT_STANDARD_DEDUCTION", "75000")
)
