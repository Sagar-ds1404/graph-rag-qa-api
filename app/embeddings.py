"""Embeddings generation module using sentence-transformers."""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

# Global cached model instance to avoid reloading on each request
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or load the singleton SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate normalized embeddings for a list of document chunk texts.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        2D numpy array of shape (N, dimension), float32, L2-normalized.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedding_model()
    # Normalize embeddings for cosine similarity with FAISS Inner Product (IndexFlatIP)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def generate_query_embedding(query: str) -> np.ndarray:
    """Generate normalized embedding for a single user question.
    
    Args:
        query: User question string.
        
    Returns:
        2D numpy array of shape (1, dimension), float32, L2-normalized.
    """
    model = get_embedding_model()
    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return np.ascontiguousarray(embedding, dtype=np.float32)
