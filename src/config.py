from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Central configuration for ML/DL Knowledge Base system."""

    # Project directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DOCUMENTS_DIR: Path = BASE_DIR / "raw_documents"
    DB_PATH: Path = DATA_DIR / "knowledge_base.db"

    # NLP & Chunking Hyperparameters (Rule #1)
    CHUNK_SIZE_WORDS: int = 300
    CHUNK_OVERLAP_WORDS: int = 50

    # Models (Rule #2)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SPACY_MODEL_NAME: str = "en_core_web_sm"

    # Search defaults
    DEFAULT_TOP_K: int = 5


# Global singleton instance
config = AppConfig()
