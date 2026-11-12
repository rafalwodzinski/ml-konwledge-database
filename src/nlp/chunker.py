from typing import List
from src.config import config


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Splits a document text into overlapping chunks of words.

    Follows Architectural Rule #1: Ensures chunks fit well within the 512 token limit
    of SentenceTransformers (all-MiniLM-L6-v2) by maintaining context through overlap.

    Args:
        text: Input document raw text.
        chunk_size: Maximum number of words per chunk (default: 300).
        overlap: Number of overlapping words between consecutive chunks (default: 50).

    Returns:
        List of text chunks. Empty list if input text is empty or whitespace-only.

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly smaller than chunk_size")

    words = text.strip().split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []

    for start_idx in range(0, len(words), step):
        end_idx = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]
        chunks.append(" ".join(chunk_words))
        if end_idx >= len(words):
            break

    return chunks

