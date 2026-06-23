from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base


def _sqlite_fallback_url() -> str:
    return f"sqlite:///{Path('data') / 'knowledge_assistant_local.db'}"


def _should_fallback_to_sqlite(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message for token in ("psycopg", "connection", "refused", "could not translate")
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    try:
        engine = create_engine(settings.database_url, future=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # pragma: no cover - exercised in integration scenarios
        if "postgresql" not in settings.database_url or not _should_fallback_to_sqlite(exc):
            raise
        fallback_engine = create_engine(_sqlite_fallback_url(), future=True)
        return fallback_engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


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
