from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from src.config import config
from src.database.repository import DatabaseRepository
from src.exact_search import ExactMatchResult, search_exact_match

# English stopwords set per scikit-learn specification
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always", "am", "among",
    "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone",
    "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became",
    "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
    "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
    "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
    "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
    "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever",
    "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty",
    "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
    "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt",
    "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
    "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in",
    "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last",
    "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile",
    "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",
    "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine",
    "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off",
    "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise",
    "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please",
    "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
    "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so",
    "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein",
    "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
    "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top",
    "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon",
    "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence",
    "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
    "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
    "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
    "yours", "yourself", "yourselves"
}


@dataclass
class SearchResult:
    """Represents a ranked retrieval result with parent document metadata."""
    document_title: str
    chunk_text: str
    score: float
    chunk_id: int
    chunk_index: int
    document_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result to dictionary format required by UI and API."""
        return {
            "document_title": self.document_title,
            "chunk_text": self.chunk_text,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
        }


class TFIDFSearchEngine:
    """TF-IDF Search Engine fitted ONCE at startup across all database chunks (Rule #3)."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.repository = DatabaseRepository(db_path or config.DB_PATH)
        self.chunk_metadata: List[Dict[str, Any]] = []
        self._sklearn_vectorizer = None
        self._sklearn_matrix = None
        self._pure_python_mode = False

        # Try initializing with scikit-learn
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._TfidfVectorizer = TfidfVectorizer
            self._cosine_similarity = cosine_similarity
        except ImportError:
            self._TfidfVectorizer = None
            self._cosine_similarity = None
            self._pure_python_mode = True

        # Fit vectorizer ONCE upon initialization
        self._fit_index()

    def _fit_index(self) -> None:
        """Fetches all chunks from SQLite and builds the TF-IDF representation."""
        self.chunk_metadata = self.repository.get_all_chunks_with_parent_info()
        if not self.chunk_metadata:
            return

        corpus = [record["chunk_text"] for record in self.chunk_metadata]

        if not self._pure_python_mode and self._TfidfVectorizer is not None:
            # Production scikit-learn implementation (Rule #3)
            self._sklearn_vectorizer = self._TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                norm="l2",
            )
            self._sklearn_matrix = self._sklearn_vectorizer.fit_transform(corpus)
        else:
            # Resilient built-in pure Python fallback
            self._build_pure_python_tfidf(corpus)

    def _build_pure_python_tfidf(self, corpus: List[str]) -> None:
        """Lightweight pure-python fallback for environments without scikit-learn installed."""
        num_docs = len(corpus)
        doc_freq: Dict[str, int] = {}
        tokenized_docs: List[List[str]] = []

        for doc in corpus:
            tokens = re.findall(r"\b[a-zA-Z]{2,}\b", doc.lower())
            filtered = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            tokenized_docs.append(filtered)
            for token in set(filtered):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self.idf: Dict[str, float] = {
            t: math.log((1 + num_docs) / (1 + df)) + 1.0
            for t, df in doc_freq.items()
        }

        self.doc_vectors: List[Dict[str, float]] = []
        for tokens in tokenized_docs:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
            total_tokens = len(tokens) or 1
            vector: Dict[str, float] = {}
            norm_sq = 0.0
            for t, count in tf.items():
                val = (count / total_tokens) * self.idf.get(t, 1.0)
                vector[t] = val
                norm_sq += val * val
            norm = math.sqrt(norm_sq) or 1.0
            self.doc_vectors.append({t: v / norm for t, v in vector.items()})

    def reload(self) -> None:
        """Refreshes the index if new documents were added to the database."""
        self._fit_index()

    def search_tfidf(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Converts user query, calculates cosine similarity and returns ranked results.

        Args:
            query: User search text.
            top_k: Number of highest ranking results to return (default: 5).

        Returns:
            List of dictionaries containing document_title, chunk_text, and TF-IDF score.
        """
        if not query or not query.strip() or not self.chunk_metadata:
            return []

        if not self._pure_python_mode and self._sklearn_vectorizer is not None:
            query_vec = self._sklearn_vectorizer.transform([query])
            similarities = self._cosine_similarity(query_vec, self._sklearn_matrix).flatten()
            ranked_indices = similarities.argsort()[::-1]

            results: List[Dict[str, Any]] = []
            for idx in ranked_indices:
                score = float(similarities[idx])
                if score <= 0.0 and len(results) >= 1:
                    break
                record = self.chunk_metadata[idx]
                results.append(
                    SearchResult(
                        document_title=record["document_title"],
                        chunk_text=record["chunk_text"],
                        score=score,
                        chunk_id=record["chunk_id"],
                        chunk_index=record["chunk_index"],
                        document_id=record["document_id"],
                    ).to_dict()
                )
                if len(results) >= top_k:
                    break
            return results

        # Pure Python fallback ranking
        query_tokens = [
            t for t in re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
            if t not in ENGLISH_STOP_WORDS
        ]
        if not query_tokens:
            return []

        q_tf: Dict[str, float] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0.0) + 1.0
        q_norm_sq = 0.0
        q_vec: Dict[str, float] = {}
        for t, count in q_tf.items():
            val = (count / len(query_tokens)) * self.idf.get(t, 1.0)
            q_vec[t] = val
            q_norm_sq += val * val
        q_norm = math.sqrt(q_norm_sq) or 1.0
        normalized_q = {t: v / q_norm for t, v in q_vec.items()}

        scored: List[tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            dot_product = sum(weight * doc_vec.get(term, 0.0) for term, weight in normalized_q.items())
            scored.append((idx, dot_product))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored:
            if score <= 0.0 and len(results) >= 1:
                break
            record = self.chunk_metadata[idx]
            results.append(
                SearchResult(
                    document_title=record["document_title"],
                    chunk_text=record["chunk_text"],
                    score=score,
                    chunk_id=record["chunk_id"],
                    chunk_index=record["chunk_index"],
                    document_id=record["document_id"],
                ).to_dict()
            )
            if len(results) >= top_k:
                break

        return results

    def search_exact(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Exact substring search across loaded chunks."""
        matches = search_exact_match(query, self.chunk_metadata)
        return matches[:top_k]

