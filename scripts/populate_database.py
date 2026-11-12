"""Script to ingest raw .txt Wikipedia articles into SQLite database with overlapping chunking."""

import argparse
from pathlib import Path
import sys

# Adjust path to import src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chunker import TextChunker
from src.config import config
from src.database.repository import DatabaseRepository


def populate_database(
    input_dir: Path = config.RAW_DOCUMENTS_DIR,
    db_path: Path = config.DB_PATH,
) -> None:
    """Reads all raw .txt files, applies overlapping chunking, and persists to SQLite."""
    txt_files = sorted(list(input_dir.glob("*.txt")))
    if not txt_files:
        print(f"Error: No .txt files found in {input_dir}. Run download script first.")
        sys.exit(1)

    print(f"Found {len(txt_files)} documents in {input_dir}")
    print(f"Target SQLite database: {db_path}\n")

    chunker = TextChunker(
        chunk_size=config.CHUNK_SIZE_WORDS,
        overlap=config.CHUNK_OVERLAP_WORDS,
    )
    repo = DatabaseRepository(db_path=db_path)

    total_chunks = 0
    for idx, file_path in enumerate(txt_files, start=1):
        raw_content = file_path.read_text(encoding="utf-8")
        title = file_path.stem.replace("_", " ").title()

        # Clean and chunk document
        cleaned_content = chunker.clean_text(raw_content)
        chunks = chunker.chunk_text(cleaned_content)

        doc_id = repo.ingest_document_with_chunks(
            title=title,
            raw_content=raw_content,
            chunks=chunks,
        )
        total_chunks += len(chunks)
        print(f"[{idx}/{len(txt_files)}] Ingested: '{title}' (ID: {doc_id}) -> {len(chunks)} chunks")

    print("\nDatabase population completed successfully!")
    print(f"Total documents: {repo.count_documents()}")
    print(f"Total chunks stored: {repo.count_chunks()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate SQLite database from raw documents.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.RAW_DOCUMENTS_DIR,
        help="Folder containing raw .txt articles",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=config.DB_PATH,
        help="Path to SQLite database",
    )
    args = parser.parse_args()
    populate_database(input_dir=args.input_dir, db_path=args.db_path)

