"""Application configuration settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Project configuration settings."""

    PROJECT_NAME: str = "Graph RAG Multi-Document Question Answering API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Hybrid Graph RAG API combining FAISS vector search, "
        "Neo4j Knowledge Graph retrieval, and Google Gemini grounding."
    )

    # Base Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    INDEX_DIR: Path = DATA_DIR / "index"

    # Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "").strip()
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "").strip()
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "").strip()
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j").strip()

    # Embedding & Vector Store Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE_WORDS: int = 800
    CHUNK_OVERLAP_WORDS: int = 100
    TOP_K_RESULTS: int = 5

    def __init__(self) -> None:
        """Ensure necessary data directories exist."""
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
