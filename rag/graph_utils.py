from __future__ import annotations

import re
from collections import Counter

from rag.config import MAX_KEYWORDS_PER_CHUNK


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "may",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "under",
    "with",
    "where",
    "which",
    "rule",
    "rules",
    "section",
    "shall",
    "form",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def extract_keywords(
    text: str,
    max_keywords: int = MAX_KEYWORDS_PER_CHUNK,
) -> list[str]:
    tokens = tokenize(text)
    filtered = [token for token in tokens if token not in STOPWORDS]
    counts = Counter(filtered)
    return [keyword for keyword, _count in counts.most_common(max_keywords)]

