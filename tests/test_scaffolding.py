from pathlib import Path
import unittest

from src.config import config
from src.database.connection import db_session, init_database
from src.nlp.chunker import chunk_text


class TestProjectScaffolding(unittest.TestCase):
    def test_config_paths(self) -> None:
        self.assertIsInstance(config.BASE_DIR, Path)
        self.assertEqual(config.CHUNK_SIZE_WORDS, 300)
        self.assertEqual(config.CHUNK_OVERLAP_WORDS, 50)

    def test_database_initialization(self) -> None:
        test_db = config.DATA_DIR / "test_init.db"
        try:
            init_database(test_db)
            with db_session(test_db) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('documents', 'chunks');"
                )
                tables = {row[0] for row in cursor.fetchall()}
                self.assertIn("documents", tables)
                self.assertIn("chunks", tables)
        finally:
            if test_db.exists():
                test_db.unlink()

    def test_chunking_overlap(self) -> None:
        sample_words = [f"word_{i}" for i in range(350)]
        text = " ".join(sample_words)
        chunks = chunk_text(text, chunk_size=300, overlap=50)

        self.assertEqual(len(chunks), 2)
        first_chunk = chunks[0].split()
        second_chunk = chunks[1].split()
        self.assertEqual(len(first_chunk), 300)
        self.assertEqual(first_chunk[-50:], second_chunk[:50])


if __name__ == "__main__":
    unittest.main()

