from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from openai import OpenAI as OpenAIClient

from rag.config import (
    BASE_DIR,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    INDEX_STORAGE_DIR,
    STORAGE_DIR,
    SYSTEM_PROMPT_PATH,
)


FALLBACK_SYSTEM_PROMPT = """You are an Indian tax-law research assistant.

Use only the supplied context when answering legal questions.
If the context is insufficient, say that clearly instead of guessing.
When you rely on a source, cite its source label exactly, such as [S1] or [S2].
Separate legal analysis from assumptions.
If the user asks a personalised tax question and important facts are missing,
state what information is needed before giving a definitive answer.
"""


load_dotenv(BASE_DIR / ".env")


def ensure_storage_dirs() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to the environment or .env file."
        )


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        return FALLBACK_SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return prompt or FALLBACK_SYSTEM_PROMPT


@lru_cache(maxsize=1)
def get_embed_model() -> OpenAIEmbedding:
    require_openai_key()
    return OpenAIEmbedding(model=DEFAULT_EMBED_MODEL)


@lru_cache(maxsize=1)
def get_llm() -> OpenAI:
    require_openai_key()
    return OpenAI(
        model=DEFAULT_LLM_MODEL,
        temperature=0.1,
        reasoning_effort="low",
        system_prompt=load_system_prompt(),
    )


@lru_cache(maxsize=1)
def get_openai_chat_client() -> OpenAIClient:
    require_openai_key()
    return OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_chat_model_name() -> str:
    require_openai_key()
    return DEFAULT_CHAT_MODEL


def configure_settings() -> None:
    ensure_storage_dirs()
    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
