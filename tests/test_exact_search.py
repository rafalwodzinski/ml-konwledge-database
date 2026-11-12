import unittest

from src.exact_search import search_exact_match


class TestExactMatchSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_chunks = [
            {
                "chunk_id": 1,
                "chunk_index": 0,
                "document_id": 10,
                "document_title": "Artificial Neural Network",
                "chunk_text": "An artificial neural network simulates biological neural networks with interconnected nodes.",
            },
            {
                "chunk_id": 2,
                "chunk_index": 1,
                "document_id": 10,
                "document_title": "Artificial Neural Network",
                "chunk_text": "Each artificial neuron receives inputs and computes an activation value.",
            },
            {
                "chunk_id": 3,
                "chunk_index": 0,
                "document_id": 20,
                "document_title": "Support Vector Machine",
                "chunk_text": "Support vector machines optimize a hyperplane boundary between classes.",
            },
        ]

    def test_exact_search_finds_keyword_neural(self) -> None:
        results = search_exact_match("neural", self.sample_chunks)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_title"], "Artificial Neural Network")
        self.assertEqual(results[0]["match_count"], 2)
        self.assertIn("biological neural networks", results[0]["chunk_text"])

    def test_exact_search_case_insensitive(self) -> None:
        results_lower = search_exact_match("neuron", self.sample_chunks)
        results_upper = search_exact_match("NEURON", self.sample_chunks)
        self.assertEqual(len(results_lower), 1)
        self.assertEqual(len(results_upper), 1)
        self.assertEqual(results_lower[0]["chunk_id"], results_upper[0]["chunk_id"])

    def test_exact_search_no_match(self) -> None:
        results = search_exact_match("quantum computing", self.sample_chunks)
        self.assertEqual(results, [])

    def test_exact_search_empty_query(self) -> None:
        self.assertEqual(search_exact_match("", self.sample_chunks), [])
        self.assertEqual(search_exact_match("   ", self.sample_chunks), [])


if __name__ == "__main__":
    unittest.main()

