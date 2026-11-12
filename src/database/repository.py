from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from src.config import config
from src.database.connection import db_session, get_db_connection, init_database


class DatabaseRepository:
    """Handles CRUD operations for documents and their overlapping chunks."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or config.DB_PATH
        init_database(self.db_path)

    def insert_document(
        self, title: str, raw_content: str, url: Optional[str] = None
    ) -> int:
        """Inserts a parent document record or returns existing ID if title exists."""
        with db_session(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (title, url, raw_content)
                VALUES (?, ?, ?)
                ON CONFLICT(title) DO UPDATE SET
                    url = excluded.url,
                    raw_content = excluded.raw_content
                RETURNING id;
                """,
                (title, url, raw_content),
            )
            row = cursor.fetchone()
            if row:
                return int(row["id"])

            # Fallback if SQLite version does not support RETURNING with ON CONFLICT
            cursor.execute("SELECT id FROM documents WHERE title = ?;", (title,))
            existing = cursor.fetchone()
            return int(existing["id"])

    def insert_chunks(self, document_id: int, chunks: List[str]) -> None:
        """Inserts text chunks for a given parent document ID."""
        with db_session(self.db_path) as conn:
            cursor = conn.cursor()
            # Clear previous chunks for this document if re-indexing
            cursor.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))

            records = [
                (document_id, idx, chunk_text)
                for idx, chunk_text in enumerate(chunks)
            ]
            cursor.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, chunk_text)
                VALUES (?, ?, ?);
                """,
                records,
            )

    def ingest_document_with_chunks(
        self,
        title: str,
        raw_content: str,
        chunks: List[str],
        url: Optional[str] = None,
    ) -> int:
        """Atomically stores parent document and all its overlapping chunks."""
        doc_id = self.insert_document(title=title, raw_content=raw_content, url=url)
        self.insert_chunks(document_id=doc_id, chunks=chunks)
        return doc_id

    def get_all_chunks_with_parent_info(self) -> List[Dict[str, Any]]:
        """Retrieves all chunks joined with their parent document title."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.chunk_index AS chunk_index,
                    c.chunk_text AS chunk_text,
                    d.id AS document_id,
                    d.title AS document_title
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.id ASC;
                """
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count_documents(self) -> int:
        """Returns total count of stored parent documents."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM documents;")
            return int(cursor.fetchone()["total"])

    def count_chunks(self) -> int:
        """Returns total count of stored chunks."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM chunks;")
            return int(cursor.fetchone()["total"])

