"""FAISS Vector Store wrapper for similarity search and metadata persistence."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector index with associated chunk metadata storage."""

    def __init__(self, index_dir: Optional[Path] = None, dimension: int = 384) -> None:
        self.index_dir = index_dir or settings.INDEX_DIR
        self.dimension = dimension
        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "metadata.json"
        
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        
        self._load_or_init()

    def _load_or_init(self) -> None:
        """Load existing index and metadata from disk, or initialize empty index."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                logger.info(f"Loading FAISS index from {self.index_file}...")
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"FAISS index loaded. Total vectors: {self.index.ntotal}, "
                    f"Metadata entries: {len(self.metadata)}"
                )
                return
            except Exception as e:
                logger.error(f"Failed to load existing FAISS index: {e}. Reinitializing.")

        # Initialize fresh IndexFlatIP for cosine similarity with normalized embeddings
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """Add new chunks and their corresponding embeddings to FAISS.
        
        Args:
            chunks: List of chunk metadata dictionaries.
            embeddings: 2D numpy array of shape (N, dimension).
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            return

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        # FAISS requires contiguous float32
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(chunks)

        self.save()

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the FAISS index for top_k most similar chunks.
        
        Args:
            query_embedding: 2D numpy array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of matching chunk dictionaries with score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        actual_k = min(top_k, self.index.ntotal)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, actual_k)
        
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["score"] = float(dist)
                results.append(item)
                
        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved FAISS index and metadata to {self.index_dir}")

    def get_total_chunks(self) -> int:
        """Get total number of chunks currently indexed."""
        return self.index.ntotal if self.index else 0

    def get_documents_summary(self) -> List[Dict[str, Any]]:
        """Group stored metadata by document to provide a summary list."""
        docs: Dict[str, Dict[str, Any]] = {}
        for item in self.metadata:
            doc_id = item.get("document_id")
            doc_name = item.get("document_name", "Unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1
        return list(docs.values())


# Global singleton instance
vector_store = FAISSVectorStore()
