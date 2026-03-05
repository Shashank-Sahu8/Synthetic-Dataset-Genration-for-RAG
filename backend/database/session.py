"""
Database engine and session factory.
Reads DATABASE_URL from the environment (set in .env).
"""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# Search upward from this file for any .env (handles running from project root
# or from backend/). Fall back to the explicit backend/.env if find_dotenv fails.
_env_file = find_dotenv(usecwd=True)
if not _env_file:
    _env_file = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_file)

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Add it to your .env file:\n"
        "DATABASE_URL=postgresql://user:password@host:5432/dbname"
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # test connections before using from pool
    pool_size=10,
    max_overflow=20,
    echo=False,               # set True to log all SQL statements
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI / Starlette dependency that yields a database session
    and guarantees cleanup even if an exception is raised.

    Usage:
        @router.get("/...")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
