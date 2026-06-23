from __future__ import annotations

from typing import Any


def format_book_citation(
    book_title: str,
    page_start: int | None,
    page_end: int | None,
    chunk_id: str,
) -> dict[str, Any]:
    if page_start and page_end and page_start != page_end:
        page_label = f"pp. {page_start}-{page_end}"
    elif page_start:
        page_label = f"p. {page_start}"
    else:
        page_label = "page unavailable"

    return normalize_citation(
        {
            "source_type": "book",
            "title": book_title,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_id": chunk_id,
            "label": f"{book_title} ({page_label})",
        }
    )


def normalize_citation(citation: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "source_type": citation.get("source_type", "unknown"),
        "title": citation.get("title", "Unknown source"),
        "page_start": citation.get("page_start"),
        "page_end": citation.get("page_end"),
        "chunk_id": citation.get("chunk_id"),
        "url": citation.get("url"),
        "label": citation.get("label") or citation.get("title", "Unknown source"),
    }
    return normalized
