from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator, Optional

from src.config import config


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates and configures an SQLite database connection.

    Args:
        db_path: Optional Path to database file. Defaults to config.DB_PATH.

    Returns:
        Configured sqlite3.Connection with foreign keys enabled and Row factory.
    """
    target_path = db_path or config.DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def db_session(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite transactions with automatic commit/rollback."""
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: Optional[Path] = None) -> None:
    """Initializes database schema from schema.sql file."""
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with schema_path.open("r", encoding="utf-8") as f:
        schema_sql = f.read()

    with db_session(db_path) as conn:
        conn.executescript(schema_sql)

