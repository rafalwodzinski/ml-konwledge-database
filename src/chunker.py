from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import uuid

from src.config import config


@dataclass
class ChunkedDocument:
    """Represents a processed document split into overlapping chunks."""
    document_id: str
    title: str
    chunks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes document and its chunks into an export-ready dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chunks": self.chunks,
            "chunk_count": len(self.chunks),
        }


class TextChunker:
    """Handles text cleaning and sliding-window overlapping chunking (Rule #1)."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE_WORDS,
        overlap: int = config.CHUNK_OVERLAP_WORDS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in range [0, chunk_size)")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def clean_text(self, text: str) -> str:
        """Cleans raw Wikipedia text by removing structural markup and metadata sections.

        - Removes boilerplate trailing sections: 'See also', 'References', 'Further reading', 'External links'
        - Removes inline section headers like '== History ==' or '=== Early work ==='
        - Normalizes citation brackets, spaces, punctuation and consecutive empty lines.
        """
        if not text:
            return ""

        # Remove trailing sections like References, See also, etc.
        trailing_sections_pattern = re.compile(
            r"\n==+\s*(See also|References|Further reading|External links|Notes)\s*==+.*",
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = trailing_sections_pattern.sub("", text)

        # Remove remaining section headers (e.g., == Overview == or === Subtitle ===)
        cleaned = re.sub(r"==+[^=\n]+==+", "", cleaned)

        # Remove citation brackets like [1], [23], [edit]
        cleaned = re.sub(r"\[\d+\]|\[edit\]", "", cleaned)

        # Remove spaces before punctuation caused by stripping tags
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

        # Normalize whitespace and excessive newlines
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)

        return cleaned.strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """Splits cleaned text into overlapping chunks using a sliding window.

        Args:
            text: Input text string.
            chunk_size: Optional custom words per chunk (defaults to self.chunk_size).
            overlap: Optional custom words overlap (defaults to self.overlap).

        Returns:
            List of chunk strings with exact word overlap on boundaries.
        """
        c_size = chunk_size or self.chunk_size
        c_overlap = overlap if overlap is not None else self.overlap
        step = c_size - c_overlap

        words = text.strip().split()
        if not words:
            return []

        if len(words) <= c_size:
            return [" ".join(words)]

        chunks: List[str] = []
        for start_idx in range(0, len(words), step):
            end_idx = start_idx + c_size
            chunk_words = words[start_idx:end_idx]
            chunks.append(" ".join(chunk_words))
            if end_idx >= len(words):
                break

        return chunks

    def chunk_document(
        self,
        raw_text: str,
        title: str,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cleans and chunks a document, returning an export-ready dictionary.

        Args:
            raw_text: Raw article content.
            title: Human-readable article title.
            document_id: Optional unique identifier. Generated if not provided.

        Returns:
            Dictionary with document_id, title, chunks, and chunk_count.
        """
        doc_id = document_id or str(uuid.uuid4())
        cleaned = self.clean_text(raw_text)
        chunks = self.chunk_text(cleaned)

        doc = ChunkedDocument(
            document_id=doc_id,
            title=title,
            chunks=chunks,
        )
        return doc.to_dict()

    def chunk_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Reads a .txt file, extracts metadata, and returns the export dictionary."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ").title()
        doc_id = path.stem

        return self.chunk_document(raw_text=raw_text, title=title, document_id=doc_id)


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Helper function for backward compatibility and quick chunking."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_text(text)
