import os
import time
import logging
from typing import Generator, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool, NullPool

from backend.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine = None
_SessionFactory = None
_last_db_check: Optional[Dict[str, Any]] = None
_last_db_check_time: float = 0.0


def get_engine(db_url: Optional[str] = None):
    """
    Returns an active SQLAlchemy engine instance with production-grade connection pooling.
    Supports PostgreSQL (QueuePool with auto-reconnection and recycling) and
    SQLite (in-memory or file-based for local development and testing).
    """
    global _engine
    target_url = db_url or settings.DATABASE_URL
    
    if _engine is not None and db_url is None:
        return _engine

    connect_args = {}
    engine_kwargs: Dict[str, Any] = {
        "echo": False,
    }

    if target_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in target_url:
            engine_kwargs["poolclass"] = StaticPool
        engine_kwargs["connect_args"] = connect_args
        engine = create_engine(target_url, **engine_kwargs)
    else:
        # PostgreSQL / Production relational database
        pool_size = getattr(settings, "DB_POOL_SIZE", 5)
        max_overflow = getattr(settings, "DB_MAX_OVERFLOW", 10)
        pool_timeout = getattr(settings, "DB_POOL_TIMEOUT", 30)
        pool_recycle = getattr(settings, "DB_POOL_RECYCLE", 1800)
        connect_timeout = getattr(settings, "DB_CONNECT_TIMEOUT", 5)

        connect_args["connect_timeout"] = connect_timeout
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": pool_timeout,
            "pool_recycle": pool_recycle,
            "pool_pre_ping": True,  # Auto-recovers from dropped/stale connections
            "connect_args": connect_args,
        })

        try:
            test_engine = create_engine(target_url, **engine_kwargs)
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine = test_engine
        except Exception as exc:
            logger.warning(
                "PostgreSQL connection failed (%s). Falling back to local SQLite.",
                type(exc).__name__
            )
            root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            db_dir = os.path.join(root_path, "data")
            os.makedirs(db_dir, exist_ok=True)
            sqlite_file = os.path.join(db_dir, "ai_healthcare.db")
            fallback_url = f"sqlite:///{sqlite_file}"
            engine = create_engine(fallback_url, connect_args={"check_same_thread": False})
            Base.metadata.create_all(bind=engine)

    if db_url is None:
        _engine = engine

    return engine


def get_session_factory(db_url: Optional[str] = None):
    """Returns the sessionmaker factory for creating database sessions."""
    global _SessionFactory
    if _SessionFactory is not None and db_url is None:
        return _SessionFactory

    engine = get_engine(db_url)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if db_url is None:
        _SessionFactory = factory
    return factory


def SessionLocal() -> Session:
    """Creates a new database session from the default configured session factory."""
    factory = get_session_factory()
    return factory()


def init_db(db_url: Optional[str] = None) -> None:
    """
    Initializes database tables by creating all schemas registered in Base.metadata.
    """
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session per request.
    Rolls back automatically on unhandled exceptions and closes cleanly.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_connection(engine=None, max_age_seconds: float = 30.0) -> Dict[str, Any]:
    """
    Non-destructive health and readiness probe for the database and connection pool.
    Verifies that the pool can check out and execute a lightweight ping (SELECT 1).
    Caches successful probe status for max_age_seconds (default 30.0s) to prevent
    high-frequency health probes from causing downstream latency degradation.
    """
    global _last_db_check, _last_db_check_time
    now = time.time()

    if engine is None and _last_db_check is not None and (now - _last_db_check_time) < max_age_seconds:
        return _last_db_check

    eng = engine or get_engine()
    dialect_name = eng.dialect.name
    pool = eng.pool

    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        pool_status = "healthy"
        pool_info = {
            "size": getattr(pool, "size", lambda: 0)(),
            "checkedin": getattr(pool, "checkedin", lambda: 0)(),
            "checkedout": getattr(pool, "checkedout", lambda: 0)(),
            "overflow": getattr(pool, "overflow", lambda: 0)(),
        }
        res = {
            "status": "ready",
            "dialect": dialect_name,
            "pool_status": pool_status,
            "pool_metrics": pool_info,
            "error": None
        }
        if engine is None:
            _last_db_check = res
            _last_db_check_time = now
        return res
    except Exception as exc:
        _last_db_check = None
        _last_db_check_time = 0.0
        return {
            "status": "degraded",
            "dialect": dialect_name,
            "pool_status": "error",
            "pool_metrics": None,
            "error": f"{type(exc).__name__}: {str(exc)}"
        }


def dispose_engine() -> None:
    """
    Disposes the active database connection pool, closing all open connections.
    Should be called upon application shutdown.
    """
    global _engine, _SessionFactory, _last_db_check, _last_db_check_time
    _last_db_check = None
    _last_db_check_time = 0.0
    if _engine is not None:
        try:
            _engine.dispose()
            logger.info("Database engine and connection pool cleanly disposed.")
        except Exception as exc:
            logger.warning("Error disposing database engine: %s", exc)
        finally:
            _engine = None
            _SessionFactory = None
