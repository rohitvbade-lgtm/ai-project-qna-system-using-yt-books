from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    try:
        engine = create_engine(settings.database_url, future=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # pragma: no cover - exercised in integration scenarios
        if not settings.database_url.startswith("postgresql"):
            raise
        raise RuntimeError(
            "Failed to connect to PostgreSQL. Start the configured database or point "
            "DATABASE_URL at a reachable instance."
        ) from exc


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        _ensure_pgvector_ready(engine)
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        _validate_pgvector_schema(engine)
        _ensure_pgvector_index(engine)


@contextmanager
def get_session() -> Session:
    init_db()
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_pgvector_ready(engine: Engine) -> None:
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:  # pragma: no cover - depends on local postgres install
        raise RuntimeError(
            "PostgreSQL is reachable, but the pgvector extension is not available. "
            "Install the extension on the target PostgreSQL instance before running the app."
        ) from exc


def _embedding_udt_name(engine: Engine) -> str | None:
    query = text(
        """
        SELECT udt_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'book_chunks'
          AND column_name = 'embedding'
        """
    )
    with engine.connect() as connection:
        return connection.execute(query).scalar_one_or_none()


def _validate_pgvector_schema(engine: Engine) -> None:
    udt_name = _embedding_udt_name(engine)
    if udt_name in {None, "vector"}:
        return

    raise RuntimeError(
        "PostgreSQL is using pgvector, but book_chunks.embedding is not a vector column. "
        "Migrate the schema manually so book_chunks.embedding uses the vector type before "
        "running the app."
    )


def _ensure_pgvector_index(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_book_chunks_embedding_hnsw
                ON book_chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
