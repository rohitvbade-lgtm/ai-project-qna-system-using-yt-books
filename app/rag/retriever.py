from __future__ import annotations

import math
import sys
import uuid
from collections import Counter
from typing import Any

from rich.console import Console
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.db.models import Book, BookChunk
from app.db.session import get_session
from app.rag.citations import format_book_citation
from app.rag.embeddings import get_embedding_provider

console = Console()


def _tokenize(text_value: str) -> list[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in text_value.split() if token.strip()]


def _keyword_score(question: str, candidate: str) -> float:
    question_counts = Counter(_tokenize(question))
    candidate_counts = Counter(_tokenize(candidate))
    if not question_counts or not candidate_counts:
        return 0.0
    overlap = sum(min(count, candidate_counts[token]) for token, count in question_counts.items())
    density_bonus = overlap / max(len(candidate_counts), 1)
    return float(overlap) + density_bonus


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(value * value for value in vector_a)) or 1.0
    norm_b = math.sqrt(sum(value * value for value in vector_b)) or 1.0
    return numerator / (norm_a * norm_b)


def _rows_to_payload(rows: list[BookChunk | dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            payload.append(row)
            continue

        payload.append(
            {
                "chunk_id": str(row.id),
                "book_title": row.book.title,
                "page_start": row.page_start,
                "page_end": row.page_end,
                "chunk_text": row.chunk_text,
                "citation": format_book_citation(
                    book_title=row.book.title,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    chunk_id=str(row.id),
                ),
            }
        )
    return payload


def retrieve_book_chunks(
    question: str,
    top_k: int | None = None,
    session: Session | None = None,
) -> list[dict[str, Any]]:
    settings = get_settings()
    effective_top_k = top_k or settings.retrieval_top_k

    def _run(active_session: Session) -> list[dict[str, Any]]:
        statement = (
            select(BookChunk)
            .options(joinedload(BookChunk.book))
            .join(BookChunk.book)
            .order_by(BookChunk.created_at.asc())
        )
        all_chunks = list(active_session.scalars(statement))
        if not all_chunks:
            return []

        chunks_with_embeddings = [chunk for chunk in all_chunks if chunk.embedding]
        provider = get_embedding_provider(allow_fake=False, require_real=False)

        if provider and chunks_with_embeddings:
            query_embedding = provider.embed_query(question)
            scored = sorted(
                chunks_with_embeddings,
                key=lambda chunk: _cosine_similarity(query_embedding, chunk.embedding or []),
                reverse=True,
            )
            return _rows_to_payload(scored[:effective_top_k])

        scored = sorted(
            all_chunks,
            key=lambda chunk: _keyword_score(question, chunk.chunk_text),
            reverse=True,
        )
        return _rows_to_payload(scored[:effective_top_k])

    if session is not None:
        return _run(session)

    with get_session() as managed_session:
        return _run(managed_session)


def get_book_chunk_by_id(chunk_id: str, session: Session | None = None) -> dict[str, Any] | None:
    def _run(active_session: Session) -> dict[str, Any] | None:
        try:
            chunk_uuid = uuid.UUID(chunk_id)
        except ValueError:
            return None
        statement = (
            select(BookChunk).options(joinedload(BookChunk.book)).where(BookChunk.id == chunk_uuid)
        )
        chunk = active_session.scalar(statement)
        if chunk is None:
            return None
        return _rows_to_payload([chunk])[0]

    if session is not None:
        return _run(session)

    with get_session() as managed_session:
        return _run(managed_session)


def list_available_books(session: Session | None = None) -> list[dict[str, Any]]:
    def _run(active_session: Session) -> list[dict[str, Any]]:
        statement = select(Book).order_by(Book.created_at.asc())
        books = list(active_session.scalars(statement))
        return [
            {
                "id": str(book.id),
                "title": book.title,
                "author": book.author,
                "file_name": book.file_name,
                "created_at": book.created_at.isoformat() if book.created_at else None,
            }
            for book in books
        ]

    if session is not None:
        return _run(session)

    with get_session() as managed_session:
        return _run(managed_session)


def _render_cli(question: str) -> int:
    results = retrieve_book_chunks(question)
    if not results:
        console.print("[yellow]No indexed library chunks found. Run ingestion first.[/yellow]")
        return 0

    table = Table(title=f"Top library chunks for: {question}")
    table.add_column("Document")
    table.add_column("Pages")
    table.add_column("Chunk")
    for item in results:
        page_start = item["page_start"]
        page_end = item["page_end"]
        pages = (
            f"{page_start}-{page_end}"
            if page_start and page_end and page_start != page_end
            else str(page_start or "?")
        )
        table.add_row(
            item["book_title"],
            pages,
            item["chunk_text"][:140] + ("..." if len(item["chunk_text"]) > 140 else ""),
        )
    console.print(table)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print('[red]Usage: uv run python -m app.rag.retriever "your question"[/red]')
        raise SystemExit(1)
    raise SystemExit(_render_cli(" ".join(sys.argv[1:])))
