from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.models import Book, BookChunk
from app.db.session import get_session
from app.rag.chunking import PageChunk, chunk_pages, clean_text
from app.rag.embeddings import MissingEmbeddingConfigurationError, get_embedding_provider
from app.runtime_logging import configure_logging, get_logger

console = Console()
app = typer.Typer(help="Ingest PDF books and documents into the local knowledge base.")
logger = get_logger(__name__)


def _count_tokens(text_value: str) -> int:
    try:
        import tiktoken
    except ImportError:
        return len(text_value.split())

    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text_value))


def extract_pages_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("PyMuPDF is not installed. Install project dependencies first.") from exc

    document = fitz.open(pdf_path)
    logger.info("ingest: reading PDF pages from %s", pdf_path.name)
    pages: list[tuple[int, str]] = []
    try:
        for page_index, page in enumerate(document, start=1):
            pages.append((page_index, clean_text(page.get_text("text"))))
    finally:
        document.close()
    return pages


def _derive_book_metadata(pdf_path: Path) -> tuple[str, str | None]:
    title = pdf_path.stem.replace("_", " ").replace("-", " ").strip().title()
    return title, None


def _upsert_book_and_chunks(
    pdf_path: Path,
    chunks: list[PageChunk],
    embeddings: list[list[float]] | None,
) -> dict[str, Any]:
    title, author = _derive_book_metadata(pdf_path)
    logger.info("ingest: storing %s with %s chunks", title, len(chunks))

    with get_session() as session:
        existing = session.scalar(select(Book).where(Book.file_name == pdf_path.name))
        if existing:
            session.execute(delete(BookChunk).where(BookChunk.book_id == existing.id))
            session.execute(delete(Book).where(Book.id == existing.id))
            session.flush()

        book = Book(title=title, author=author, file_name=pdf_path.name)
        session.add(book)
        session.flush()

        for chunk in chunks:
            embedding = embeddings[chunk.chunk_index] if embeddings else None
            session.add(
                BookChunk(
                    book_id=book.id,
                    chunk_index=chunk.chunk_index,
                    chapter=None,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    chunk_text=chunk.text,
                    token_count=_count_tokens(chunk.text),
                    embedding=embedding,
                )
            )

        return {
            "book_id": str(book.id),
            "title": title,
            "chunks_indexed": len(chunks),
        }


def ingest_pdf(pdf_path: Path) -> dict[str, Any]:
    settings = get_settings()
    logger.info("ingest: processing %s", pdf_path.name)
    pages = extract_pages_from_pdf(pdf_path)
    logger.info("ingest: extracted %s pages", len(pages))
    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    logger.info("ingest: created %s chunks", len(chunks))

    if not chunks:
        return {
            "file_name": pdf_path.name,
            "chunks_indexed": 0,
            "warning": "No extractable text found.",
        }

    provider = get_embedding_provider(allow_fake=False, require_real=False)
    embeddings: list[list[float]] | None = None
    warning: str | None = None

    if provider is not None:
        logger.info("ingest: generating embeddings for %s chunks", len(chunks))
        embeddings = provider.embed_texts([chunk.text for chunk in chunks])
    else:
        logger.info("ingest: no embedding provider enabled; storing chunks without embeddings")
        warning = (
            "No embedding provider is enabled. Stored chunks without embeddings; "
            "retrieval will use keyword fallback."
        )

    result = _upsert_book_and_chunks(pdf_path, chunks, embeddings)
    if warning:
        result["warning"] = warning
    return result


def ingest_books_directory(raw_dir: Path | None = None) -> list[dict[str, Any]]:
    raw_directory = raw_dir or Path("data/books/raw")
    if not raw_directory.exists():
        raw_directory.mkdir(parents=True, exist_ok=True)
        return []

    logger.info("ingest: scanning %s for PDF files", raw_directory)
    results: list[dict[str, Any]] = []
    for pdf_path in sorted(raw_directory.glob("*.pdf")):
        try:
            results.append(ingest_pdf(pdf_path))
        except MissingEmbeddingConfigurationError as exc:
            logger.exception("ingest: embedding configuration failed for %s", pdf_path.name)
            results.append({"file_name": pdf_path.name, "chunks_indexed": 0, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - integration failure path
            logger.exception("ingest: ingestion failed for %s", pdf_path.name)
            results.append({"file_name": pdf_path.name, "chunks_indexed": 0, "error": str(exc)})
    return results


@app.command("run")
def ingest_books_cli() -> None:
    configure_logging()
    results = ingest_books_directory()
    if not results:
        console.print("[yellow]No PDF files found in data/books/raw.[/yellow]")
        return

    console.print_json(json=json.dumps(results))


if __name__ == "__main__":
    app()
