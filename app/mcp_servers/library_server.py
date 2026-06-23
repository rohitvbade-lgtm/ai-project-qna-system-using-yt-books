from __future__ import annotations

from typing import Any

from app.rag.retriever import (
    get_book_chunk_by_id as retrieve_book_chunk_by_id,
)
from app.rag.retriever import (
    list_available_books as retrieve_available_books,
)
from app.rag.retriever import (
    retrieve_book_chunks,
)


def search_book_chunks(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    return retrieve_book_chunks(query, top_k=top_k)


def get_book_chunk_by_id(chunk_id: str) -> dict[str, Any] | None:
    return retrieve_book_chunk_by_id(chunk_id)


def list_available_books() -> list[dict[str, Any]]:
    return retrieve_available_books()


def build_server() -> Any:
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("fastmcp is not installed. Install project dependencies first.") from exc

    server = FastMCP("knowledge-library-tools")
    server.tool(name="search_book_chunks")(search_book_chunks)
    server.tool(name="get_book_chunk_by_id")(get_book_chunk_by_id)
    server.tool(name="list_available_books")(list_available_books)
    return server


def main() -> None:
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
