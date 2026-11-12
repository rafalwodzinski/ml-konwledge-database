from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional


@dataclass
class ExactMatchResult:
    """Represents an exact keyword match result with parent document metadata."""
    document_title: str
    chunk_text: str
    match_count: int
    chunk_id: Optional[int]
    chunk_index: int
    document_id: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to dictionary format for Streamlit rendering."""
        return {
            "document_title": self.document_title,
            "chunk_text": self.chunk_text,
            "match_count": self.match_count,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
        }


def search_exact_match(
    query: str,
    chunks: List[Dict[str, Any]],
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """Finds all chunks containing the exact substring query (Learn as I go: in text.lower()).

    Args:
        query: Target keyword or phrase.
        chunks: Collection of chunk records with 'chunk_text' and 'document_title'.
        case_sensitive: Whether the search respects case. Defaults to False.

    Returns:
        List of matching chunk dictionaries ranked by occurrence count.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    target = cleaned_query if case_sensitive else cleaned_query.lower()
    results: List[ExactMatchResult] = []

    for chunk in chunks:
        raw_text = chunk.get("chunk_text", "")
        text_to_search = raw_text if case_sensitive else raw_text.lower()

        if target in text_to_search:
            # Count occurrences using regex or simple count
            escaped = re.escape(target)
            flags = 0 if case_sensitive else re.IGNORECASE
            count = len(re.findall(escaped, raw_text, flags=flags)) or 1

            results.append(
                ExactMatchResult(
                    document_title=chunk.get("document_title", "Uploaded Document"),
                    chunk_text=raw_text,
                    match_count=count,
                    chunk_id=chunk.get("chunk_id"),
                    chunk_index=chunk.get("chunk_index", 0),
                    document_id=chunk.get("document_id"),
                )
            )

    # Sort primarily by match count descending
    results.sort(key=lambda r: r.match_count, reverse=True)
    return [r.to_dict() for r in results]

