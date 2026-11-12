"""Verification script for Definition of Done (Issue #1).

Verifies that for an article, edge words from the end of chunk 1
are strictly equal to the words at the beginning of chunk 2.
"""

from pathlib import Path
import random
import sys

# Adjust path to include src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.chunker import TextChunker
from src.config import config


def main() -> None:
    raw_dir = config.RAW_DOCUMENTS_DIR
    txt_files = list(raw_dir.glob("*.txt"))

    chunker = TextChunker(
        chunk_size=config.CHUNK_SIZE_WORDS,
        overlap=config.CHUNK_OVERLAP_WORDS,
    )

    if not txt_files:
        print(f"No .txt documents found in {raw_dir}.")
        print("Generating a test ML/DL document to verify chunking overlap DoD...")
        sample_path = raw_dir / "sample_transformer_architecture.txt"
        words = [f"token_{i}" for i in range(750)]
        sample_content = (
            "== Introduction ==\n"
            "Transformers rely on self-attention mechanisms to compute representations.\n\n"
            + " ".join(words)
            + "\n\n== References ==\nVaswani et al. (2017) Attention Is All You Need."
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(sample_content, encoding="utf-8")
        txt_files = [sample_path]

    selected_file = random.choice(txt_files)
    print(f"--> Verifying article: '{selected_file.name}'")

    doc_dict = chunker.chunk_file(selected_file)
    chunks = doc_dict["chunks"]

    print(f"--> Document ID: {doc_dict['document_id']}")
    print(f"--> Title: {doc_dict['title']}")
    print(f"--> Total generated chunks: {len(chunks)}")

    if len(chunks) < 2:
        print(f"Document has only {len(chunks)} chunk. Needs at least 2 chunks to test overlap.")
        sys.exit(1)

    chunk_1_words = chunks[0].split()
    chunk_2_words = chunks[1].split()

    overlap_size = config.CHUNK_OVERLAP_WORDS
    tail_chunk_1 = chunk_1_words[-overlap_size:]
    head_chunk_2 = chunk_2_words[:overlap_size]

    print(f"\n[Validation] Overlap size target: {overlap_size} words")
    print(f"End of Chunk 1 ({overlap_size} words): {' '.join(tail_chunk_1[:5])} ... {' '.join(tail_chunk_1[-5:])}")
    print(f"Start of Chunk 2 ({overlap_size} words): {' '.join(head_chunk_2[:5])} ... {' '.join(head_chunk_2[-5:])}")

    assert tail_chunk_1 == head_chunk_2, "Overlap mismatch between Chunk 1 and Chunk 2!"

    print("\n[SUCCESS] Definition of Done satisfied:")
    print("Edge words from the end of Chunk 1 strictly match the start of Chunk 2!")


if __name__ == "__main__":
    main()

