"""SQLAlchemy engine and session lifecycle helpers."""

from collections.abc import Iterator
import os
from threading import Lock

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from backend.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_order_read_engine: Engine | None = None
_order_read_session_factory: sessionmaker[Session] | None = None
_initialization_lock = Lock()


def _database_url_from_env() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it to a PostgreSQL connection URL, "
            "for example postgresql+psycopg://user:password@localhost:5432/automous_return."
        )
    return database_url


def create_engine_from_url(database_url: str) -> Engine:
    """Create an engine from a URL, selecting psycopg for PostgreSQL URLs."""
    url = make_url(database_url)
    if url.drivername in {"postgres", "postgresql", "postgresql+psycopg2"}:
        url = url.set(drivername="postgresql+psycopg")
    return create_engine(url, pool_pre_ping=True)


def get_engine() -> Engine:
    """Return the process-wide engine, creating it from DATABASE_URL on first use."""
    global _engine
    if _engine is None:
        with _initialization_lock:
            if _engine is None:
                _engine = create_engine_from_url(_database_url_from_env())
    return _engine


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Create a session factory bound to the provided or configured engine."""
    return sessionmaker(
        bind=engine if engine is not None else get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )


def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        with _initialization_lock:
            if _session_factory is None:
                _session_factory = create_session_factory(engine)
    return _session_factory


def get_order_read_session_factory() -> sessionmaker[Session]:
    """Return a session factory using the dedicated order read-only DB URL.

    ``ORDER_READ_DATABASE_URL`` must identify a database principal granted
    SELECT access only to the order-history tables. No write-capable fallback
    to ``DATABASE_URL`` is intentional.
    """
    global _order_read_engine, _order_read_session_factory
    if _order_read_session_factory is None:
        with _initialization_lock:
            if _order_read_session_factory is None:
                database_url = os.getenv("ORDER_READ_DATABASE_URL")
                if not database_url:
                    raise RuntimeError(
                        "ORDER_READ_DATABASE_URL is not configured. Configure it with a "
                        "database principal that has SELECT-only access to order data."
                    )
                _order_read_engine = create_engine_from_url(database_url)
                _order_read_session_factory = create_session_factory(_order_read_engine)
    return _order_read_session_factory


def get_db() -> Iterator[Session]:
    """Yield a session for a request; transaction commit remains the caller's job."""
    with get_session_factory()() as session:
        yield session


def init_db(engine: Engine | None = None) -> None:
    """Create registered model tables; use migrations for production schema changes."""
    target_engine = engine if engine is not None else get_engine()
    Base.metadata.create_all(target_engine)
