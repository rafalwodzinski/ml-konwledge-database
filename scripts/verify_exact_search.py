"""Verification script for Definition of Done (Issue #3).

Verifies that exact keyword search for 'neural' finds correct occurrences
across SQLite database chunks without environmental crashes.
"""

from pathlib import Path
import sys

# Adjust path to include src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import config
from src.database.repository import DatabaseRepository
from src.exact_search import search_exact_match


def main() -> None:
    repo = DatabaseRepository(config.DB_PATH)
    all_chunks = repo.get_all_chunks_with_parent_info()

    print(f"--> Database: {config.DB_PATH}")
    print(f"--> Total database chunks to search: {len(all_chunks)}")

    if not all_chunks:
        print("Error: No chunks in database! Run 'python3 scripts/populate_database.py' first.")
        sys.exit(1)

    query = "neural"
    print(f"--> Executing Exact Match search for keyword: '{query}'")

    results = search_exact_match(query, all_chunks)

    print(f"--> Found {len(results)} chunks matching exact keyword '{query}'.\n")

    if not results:
        print(f"Error: Expected matches for '{query}', but none were found!")
        sys.exit(1)

    print("=" * 80)
    print(f"EXACT MATCH RESULTS FOR: '{query}' (Top 5 preview)")
    print("=" * 80)

    for rank, res in enumerate(results[:5], start=1):
        print(f"#{rank} | Document Title: {res['document_title']}")
        print(f"   | Occurrences: {res['match_count']} | Chunk ID: {res['chunk_id']} (Index: {res['chunk_index']})")
        preview = res["chunk_text"][:140].replace("\n", " ") + "..."
        print(f"   | Snippet: {preview}")
        print("-" * 80)

    # Verification of DoD
    top_result = results[0]
    assert query.lower() in top_result["chunk_text"].lower(), "Query must be in chunk text"
    assert top_result["document_title"], "Document title must be present from SQL join"
    assert top_result["match_count"] >= 1, "Match count must be at least 1"

    print("\n[SUCCESS] Definition of Done satisfied:")
    print(f"Successfully located exact keyword '{query}' across multiple articles without runtime errors.")


if __name__ == "__main__":
    main()

