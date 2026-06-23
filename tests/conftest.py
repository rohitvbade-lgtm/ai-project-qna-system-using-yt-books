from __future__ import annotations

import uuid

import pytest


@pytest.fixture(autouse=True)
def configure_test_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    from app.config import get_settings
    from app.db.session import get_engine, get_session_factory

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    yield
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


@pytest.fixture
def seeded_library():
    from app.db.models import Book, BookChunk
    from app.db.session import get_session

    book_id = uuid.uuid4()
    with get_session() as session:
        book = Book(
            id=book_id,
            title="General Science Reader",
            author="Test Author",
            file_name="science-reader.pdf",
        )
        session.add(book)
        session.flush()
        session.add_all(
            [
                BookChunk(
                    book_id=book_id,
                    chunk_index=0,
                    page_start=8,
                    page_end=9,
                    chunk_text=(
                        "Photosynthesis converts light energy into chemical energy. "
                        "Plants use sunlight, water, and carbon dioxide to produce glucose "
                        "and release oxygen."
                    ),
                    token_count=22,
                    embedding=None,
                ),
                BookChunk(
                    book_id=book_id,
                    chunk_index=1,
                    page_start=30,
                    page_end=31,
                    chunk_text=(
                        "Plate tectonics explains how Earth's lithosphere is divided into "
                        "moving plates that interact at convergent, divergent, and transform "
                        "boundaries."
                    ),
                    token_count=21,
                    embedding=None,
                ),
            ]
        )
    return True
