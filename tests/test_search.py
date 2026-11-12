from pathlib import Path
import tempfile
import unittest

from src.database.repository import DatabaseRepository
from src.search import TFIDFSearchEngine


class TestSearchAndDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_search.db"
        self.repo = DatabaseRepository(self.db_path)

        # Ingest sample documents and chunks
        self.repo.ingest_document_with_chunks(
            title="Transformer Architecture",
            raw_content="Transformer models use self-attention mechanisms for language processing.",
            chunks=[
                "Transformer models rely on self-attention mechanisms to process tokens in parallel.",
                "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            ],
        )

        self.repo.ingest_document_with_chunks(
            title="Support Vector Machine",
            raw_content="Support vector machines are supervised learning models with associated learning algorithms.",
            chunks=[
                "Support vector machines construct a hyperplane or set of hyperplanes in a high-dimensional space.",
                "The kernel trick allows SVMs to perform non-linear classification efficiently.",
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_database_foreign_key_and_join(self) -> None:
        records = self.repo.get_all_chunks_with_parent_info()
        self.assertEqual(len(records), 4)

        for rec in records:
            self.assertIn("chunk_id", rec)
            self.assertIn("chunk_text", rec)
            self.assertIn("document_id", rec)
            self.assertIn("document_title", rec)
            self.assertIn(rec["document_title"], ["Transformer Architecture", "Support Vector Machine"])

    def test_tfidf_search_ranking(self) -> None:
        engine = TFIDFSearchEngine(self.db_path)

        # Search for transformer query
        results = engine.search_tfidf("self-attention transformer", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["document_title"], "Transformer Architecture")
        self.assertIn("self-attention", results[0]["chunk_text"].lower())
        self.assertGreater(results[0]["score"], 0.0)

        # Search for SVM query
        svm_results = engine.search_tfidf("kernel trick classification", top_k=1)
        self.assertGreater(len(svm_results), 0)
        self.assertEqual(svm_results[0]["document_title"], "Support Vector Machine")
        self.assertIn("kernel", svm_results[0]["chunk_text"].lower())

    def test_stop_words_elimination(self) -> None:
        engine = TFIDFSearchEngine(self.db_path)
        # Query with only stop words should yield no results or score 0
        results = engine.search_tfidf("the and or is in of")
        self.assertEqual(len(results), 0)

    def test_empty_query(self) -> None:
        engine = TFIDFSearchEngine(self.db_path)
        self.assertEqual(engine.search_tfidf(""), [])
        self.assertEqual(engine.search_tfidf("   "), [])


if __name__ == "__main__":
    unittest.main()

