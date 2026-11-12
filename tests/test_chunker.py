from pathlib import Path
import tempfile
import unittest

from src.chunker import TextChunker, chunk_text


class TestTextChunker(unittest.TestCase):
    def setUp(self) -> None:
        self.chunker = TextChunker(chunk_size=300, overlap=50)

    def test_clean_text_removes_wiki_markup_and_sections(self) -> None:
        raw_text = (
            "Machine learning is a field of study.\n\n"
            "== History ==\n"
            "Early algorithms appeared in the 1950s [1].\n\n"
            "=== Subfield ===\n"
            "Deep learning is a subset of machine learning.\n\n"
            "== See also ==\n"
            "Artificial intelligence\n\n"
            "== References ==\n"
            "1. Turing, Alan (1950).\n"
        )
        cleaned = self.chunker.clean_text(raw_text)

        self.assertNotIn("== History ==", cleaned)
        self.assertNotIn("=== Subfield ===", cleaned)
        self.assertNotIn("== See also ==", cleaned)
        self.assertNotIn("== References ==", cleaned)
        self.assertNotIn("Turing, Alan", cleaned)
        self.assertNotIn("[1]", cleaned)
        self.assertIn("Machine learning is a field of study.", cleaned)
        self.assertIn("Early algorithms appeared in the 1950s.", cleaned)

    def test_chunking_sliding_window_overlap(self) -> None:
        # Generate 650 distinct words
        words = [f"word_{i:04d}" for i in range(650)]
        text = " ".join(words)

        chunks = self.chunker.chunk_text(text)
        # 650 words:
        # Chunk 1: [0:300] (len 300)
        # Chunk 2: [250:550] (len 300)
        # Chunk 3: [500:650] (len 150)
        self.assertEqual(len(chunks), 3)

        chunk_1_words = chunks[0].split()
        chunk_2_words = chunks[1].split()
        chunk_3_words = chunks[2].split()

        self.assertEqual(len(chunk_1_words), 300)
        self.assertEqual(len(chunk_2_words), 300)
        self.assertEqual(len(chunk_3_words), 150)

        # Definition of Done: last 50 words of chunk 1 == first 50 words of chunk 2
        self.assertEqual(chunk_1_words[-50:], chunk_2_words[:50])
        self.assertEqual(chunk_2_words[-50:], chunk_3_words[:50])

    def test_chunk_document_export_dict(self) -> None:
        words = [f"token_{i}" for i in range(400)]
        raw_text = "== Overview ==\n" + " ".join(words)

        doc_dict = self.chunker.chunk_document(
            raw_text=raw_text,
            title="Gradient Descent",
            document_id="doc_gd_001",
        )

        self.assertEqual(doc_dict["document_id"], "doc_gd_001")
        self.assertEqual(doc_dict["title"], "Gradient Descent")
        self.assertIn("chunks", doc_dict)
        self.assertEqual(doc_dict["chunk_count"], 2)
        self.assertIsInstance(doc_dict["chunks"], list)

    def test_chunk_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "recurrent_neural_network.txt"
            sample_content = "== Introduction ==\n" + " ".join(["neuron"] * 350)
            file_path.write_text(sample_content, encoding="utf-8")

            res = self.chunker.chunk_file(file_path)
            self.assertEqual(res["title"], "Recurrent Neural Network")
            self.assertEqual(res["document_id"], "recurrent_neural_network")
            self.assertEqual(res["chunk_count"], 2)


if __name__ == "__main__":
    unittest.main()

