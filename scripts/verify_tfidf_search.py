"""Verification script for Definition of Done (Issue #2).

Tests that searching for 'deep learning' finds and ranks related results
at top positions, with original parent document title correctly retrieved from SQLite.
"""

import argparse
from pathlib import Path
import sys

# Adjust path to include src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import config
from src.search import TFIDFSearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TF-IDF search DoD.")
    parser.add_argument(
        "--query",
        type=str,
        default="deep learning",
        help="Search query to test (default: 'deep learning')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display",
    )
    args = parser.parse_args()

    print(f"--> Initializing TF-IDF Search Engine on database: {config.DB_PATH}")
    searcher = TFIDFSearchEngine()

    print(f"--> Total indexed chunks: {len(searcher.chunk_metadata)}")
    if not searcher.chunk_metadata:
        print("Error: No chunks found in database. Run 'python3 scripts/populate_database.py' first.")
        sys.exit(1)

    print(f"--> Executing query: '{args.query}' (top_k={args.top_k})\n")
    results = searcher.search_tfidf(args.query, top_k=args.top_k)

    if not results:
        print(f"Error: No results found for query '{args.query}'!")
        sys.exit(1)

    print("=" * 80)
    print(f"SEARCH RESULTS FOR: '{args.query}'")
    print("=" * 80)

    for rank, res in enumerate(results, start=1):
        print(f"#{rank} | Document Title: {res['document_title']}")
        print(f"   | Score: {res['score']:.4f} | Chunk ID: {res['chunk_id']} (Index: {res['chunk_index']})")
        preview = res["chunk_text"][:160].replace("\n", " ") + "..."
        print(f"   | Snippet: {preview}")
        print("-" * 80)

    # Verification of DoD
    top_doc_title = results[0]["document_title"].lower()
    top_score = results[0]["score"]

    print("\n[DoD Validation]")
    print(f"Top result parent document: '{results[0]['document_title']}'")
    print(f"Top TF-IDF score: {top_score}")

    assert top_score > 0.0, "Top score must be positive"
    assert "deep learning" in top_doc_title or "neural network" in top_doc_title or "machine learning" in top_doc_title, (
        f"Expected relevant ML/DL topic in top document, got: {results[0]['document_title']}"
    )
    assert results[0]["chunk_text"], "Chunk text must not be empty"
    assert results[0]["document_id"] > 0, "Parent document ID must be valid from SQL join"

    print("\n[SUCCESS] Definition of Done satisfied:")
    print("Found and ranked relevant ML/DL articles on top positions without missing parent document errors!")


if __name__ == "__main__":
    main()

