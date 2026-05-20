from __future__ import annotations

import re
from pathlib import Path


SECTION_PATTERN = re.compile(r"\bSection\s+\d+[A-Z]*(?:\([^)]+\))?\b", re.IGNORECASE)
RULE_PATTERN = re.compile(r"\bRule\s+\d+[A-Z]*(?:\([^)]+\))?\b", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"\bChapter\s+[IVXLC]+\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"



def infer_document_type(file_path: str | Path) -> str:
    path = Path(file_path)
    lower_name = path.name.lower()

    if "rule" in lower_name:
        return "income_tax_rules"
    if "act" in lower_name:
        return "income_tax_act"
    if "circular" in lower_name or "notification" in lower_name:
        return "cbdt_circular_or_notification"
    return "tax_reference"


def extract_section_title(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if lines:
        first_line = lines[0]
        title_candidate = re.split(r"\s+\d+\.\s*(?:\(\d+\))?", first_line, maxsplit=1)[0]
        title_candidate = title_candidate.strip(" :-")
        if 5 <= len(title_candidate) <= 140 and not title_candidate.startswith("("):
            return title_candidate

    for pattern in (SECTION_PATTERN, RULE_PATTERN, CHAPTER_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group(0)

    for line in lines[:5]:
        if len(line) <= 120:
            return line

    return "Unknown"


def extract_statute_reference(text: str) -> str:
    matches: list[str] = []
    for pattern in (SECTION_PATTERN, RULE_PATTERN, CHAPTER_PATTERN):
        match = pattern.search(text)
        if match:
            matches.append(match.group(0))

    return " | ".join(matches) if matches else "Unknown"


def extract_document_year(file_name: str) -> str:
    match = YEAR_PATTERN.search(file_name)
    return match.group(0) if match else "Unknown"


def enrich_metadata(node, idx: int) -> dict[str, str | int]:
    original_metadata = node.metadata or {}
    file_name = original_metadata.get("file_name", "Unknown")
    file_path = original_metadata.get("file_path", file_name)
    page_number = (
        original_metadata.get("page_label")
        or original_metadata.get("page_number")
        or original_metadata.get("source")
        or "Unknown"
    )
    total_pages = original_metadata.get("total_pages", "Unknown")

    document_id = _slugify(Path(file_name).stem)
    section_title = extract_section_title(node.text)
    statute_reference = extract_statute_reference(node.text)
    chunk_id = f"{document_id}-p{page_number}-c{idx:05d}"

    return {
        "source_file": file_name,
        "source_path": str(file_path),
        "document_id": document_id,
        "document_type": infer_document_type(file_path),
        "document_year": extract_document_year(file_name),
        "section_title": section_title,
        "statute_reference": statute_reference,
        "page_number": str(page_number),
        "total_pages": str(total_pages),
        "chunk_index": idx,
        "chunk_id": chunk_id,
    }
