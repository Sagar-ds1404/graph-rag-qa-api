"""Hybrid Retrieval: Combines FAISS Vector Similarity with Neo4j Knowledge Graph."""

import logging
import re
from typing import Any, Dict, List, Set, Tuple
from app.embeddings import generate_query_embedding
from app.vector_store import vector_store
from app.graph_store import graph_store

logger = logging.getLogger(__name__)


def extract_query_entities(question: str) -> List[str]:
    """Extract candidate entity names or technical terms from a user question."""
    # Look for capitalized words, alphanumeric tokens, or words >= 3 chars
    tokens = re.findall(r"\b[A-Za-z0-9_\-\.]{3,}\b", question)
    
    # Filter common question words / English stopwords
    stopwords = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "the", "and", "or", "for", "with", "from", "about", "into", "through",
        "are", "was", "were", "been", "being", "have", "has", "had", "does",
        "did", "can", "could", "should", "would", "will", "shall", "main",
        "discussed", "mentioned", "document", "documents", "tell", "explain",
        "give", "summary", "list", "show", "describe", "find", "between"
    }

    candidates: List[str] = []
    seen: Set[str] = set()

    for token in tokens:
        clean = token.strip()
        lower = clean.lower()
        if lower not in stopwords and lower not in seen:
            seen.add(lower)
            candidates.append(clean)

    return candidates


def retrieve_context(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Perform hybrid retrieval combining FAISS vector search and Neo4j graph traversal.
    
    Returns:
        Dictionary with:
        - vector_context: List of retrieved chunks from FAISS
        - graph_context: List of entities and connections from Neo4j
        - sources: Deduplicated list of source references (document, chunk_id, page)
    """
    # 1. Vector Retrieval
    vector_context: List[Dict[str, Any]] = []
    try:
        q_emb = generate_query_embedding(question)
        faiss_results = vector_store.similarity_search(q_emb, top_k=top_k)
        for r in faiss_results:
            vector_context.append({
                "text": r.get("text", ""),
                "document": r.get("document_name", "Unknown"),
                "chunk_id": r.get("chunk_id", ""),
                "page": r.get("page", 1),
                "score": r.get("score", 0.0),
            })
    except Exception as e:
        logger.error(f"Error during vector retrieval: {e}")

    # 2. Graph Retrieval
    graph_context: List[Dict[str, Any]] = []
    graph_chunks: List[Dict[str, Any]] = []
    try:
        candidates = extract_query_entities(question)
        if candidates and graph_store.is_available():
            graph_results = graph_store.search_graph_for_entities(candidates)
            for g in graph_results:
                graph_context.append({
                    "entity": g.get("entity", ""),
                    "related_entities": g.get("related_entities", []),
                })
                # Collect any chunks attached to graph entities
                for c in g.get("chunks", []):
                    if c.get("chunk_id"):
                        graph_chunks.append({
                            "document": c.get("doc_name", "Unknown"),
                            "chunk_id": c.get("chunk_id"),
                            "page": c.get("page", 1),
                            "text": c.get("text", ""),
                        })
    except Exception as e:
        logger.error(f"Error during graph retrieval: {e}")

    # 3. Combine and Deduplicate Sources
    sources: List[Dict[str, Any]] = []
    seen_sources: Set[Tuple[str, str, int]] = set()

    for item in vector_context:
        doc = item.get("document", "Unknown")
        chunk_id = item.get("chunk_id", "")
        page = item.get("page", 1)
        key = (doc, chunk_id, page)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "document": doc,
                "chunk_id": chunk_id,
                "page": page,
            })

    for item in graph_chunks:
        doc = item.get("document", "Unknown")
        chunk_id = item.get("chunk_id", "")
        page = item.get("page", 1)
        key = (doc, chunk_id, page)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "document": doc,
                "chunk_id": chunk_id,
                "page": page,
            })

    return {
        "vector_context": vector_context,
        "graph_context": graph_context,
        "sources": sources,
    }
