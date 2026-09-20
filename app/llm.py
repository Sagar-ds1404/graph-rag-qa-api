"""Gemini LLM integration using official google-genai SDK for entity extraction and grounded QA."""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)

# Global client cache
_client: Optional[genai.Client] = None


def get_genai_client() -> Optional[genai.Client]:
    """Get or initialize the Google GenAI client."""
    global _client
    if not settings.GEMINI_API_KEY:
        return None
    if _client is None:
        try:
            _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Client: {e}")
            return None
    return _client


def is_gemini_configured() -> bool:
    """Check if Gemini API key is configured."""
    return bool(settings.GEMINI_API_KEY.strip())


def fallback_entity_extraction(text: str) -> Dict[str, Any]:
    """Simple rule-based fallback when Gemini is unavailable.
    
    Identifies capitalized keywords, acronyms, and common terms.
    """
    # Find sequences of capitalized words or acronyms (e.g. "FastAPI", "Neo4j", "Vector Search")
    matches = re.findall(r"\b[A-Z][a-zA-Z0-9_\-\.]{1,25}\b", text)
    # Stopwords filter
    stopwords = {
        "The", "This", "That", "These", "Those", "And", "Or", "If", "When",
        "Where", "Why", "How", "What", "Who", "In", "On", "At", "To", "For",
        "With", "From", "By", "About", "As", "Into", "Like", "Through", "After",
        "Over", "Between", "Out", "Against", "During", "Without", "Before", "Under",
        "Around", "Among", "Note", "Figure", "Table", "Page", "Section", "Chapter"
    }
    
    entities = []
    seen = set()
    for m in matches:
        m_strip = m.strip()
        if len(m_strip) > 2 and m_strip not in stopwords and m_strip.lower() not in seen:
            seen.add(m_strip.lower())
            entities.append(m_strip)
            if len(entities) >= 8:
                break

    # Build simple sequential co-occurrence relations between consecutive entities
    relationships = []
    for i in range(len(entities) - 1):
        relationships.append({
            "source": entities[i],
            "relationship": "ASSOCIATED_WITH",
            "target": entities[i + 1],
        })

    return {
        "entities": entities,
        "relationships": relationships
    }


def extract_entities_and_relations(text: str) -> Dict[str, Any]:
    """Extract key entities and relationships from text using Gemini or fallback."""
    client = get_genai_client()
    if not client:
        return fallback_entity_extraction(text)

    prompt = f"""
Analyze the following text and extract up to 6 key technical entities (e.g. concepts, tools, frameworks, components) and relationships between them.

Text snippet:
\"\"\"{text[:2000]}\"\"\"

Return ONLY a valid JSON object matching this schema:
{{
    "entities": ["Entity1", "Entity2"],
    "relationships": [
        {{"source": "Entity1", "relationship": "USES", "target": "Entity2"}}
    ]
}}
"""

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        if response and response.text:
            data = json.loads(response.text.strip())
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            if isinstance(entities, list) and isinstance(relationships, list):
                return {
                    "entities": [str(e).strip() for e in entities if str(e).strip()],
                    "relationships": relationships,
                }
    except Exception as e:
        logger.warning(f"Gemini entity extraction failed: {e}. Using fallback.")

    return fallback_entity_extraction(text)


def generate_grounded_answer(
    question: str,
    vector_context: List[Dict[str, Any]],
    graph_context: List[Dict[str, Any]],
) -> str:
    """Generate a strictly grounded response using Gemini."""
    client = get_genai_client()
    if not client:
        raise RuntimeError(
            "Gemini API is not configured. Please set GEMINI_API_KEY in your environment or .env file."
        )

    # Format vector context
    vector_text_parts = []
    for idx, item in enumerate(vector_context, 1):
        doc = item.get("document", "Unknown")
        page = item.get("page", 1)
        chunk_id = item.get("chunk_id", f"c{idx}")
        txt = item.get("text", "")
        vector_text_parts.append(f"[Source {idx} | Doc: {doc} | Page: {page} | Chunk: {chunk_id}]\n{txt}")

    vector_formatted = "\n\n".join(vector_text_parts) if vector_text_parts else "No relevant document chunks found."

    # Format graph context
    graph_text_parts = []
    for g in graph_context:
        ent = g.get("entity", "")
        rel = ", ".join(g.get("related_entities", []))
        graph_text_parts.append(f"- Entity: {ent} -> Connected to: [{rel}]")

    graph_formatted = "\n".join(graph_text_parts) if graph_text_parts else "No knowledge graph connections found."

    system_instruction = (
        "You are a document question-answering assistant.\n"
        "Answer only using the retrieved context provided below.\n"
        "Do not invent facts or extrapolate beyond the provided text.\n"
        "If the retrieved context does not contain enough information to answer the question, "
        "clearly state: 'The provided documents do not contain sufficient information to answer this question.'\n"
        "Give a concise, factual, and well-structured answer.\n"
        "Mention which source documents/pages support your answer."
    )

    user_prompt = f"""
CONTEXT INFORMATION:

--- VECTOR RETRIEVED TEXT (FROM DOCUMENTS) ---
{vector_formatted}

--- KNOWLEDGE GRAPH RETRIEVED CONNECTIONS ---
{graph_formatted}

---------------------------------------------
QUESTION:
{question}

Grounded Answer:
"""

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        if response and response.text:
            return response.text.strip()
        return "No answer could be generated from the model."
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        raise RuntimeError(f"Error calling Gemini LLM: {str(e)}")
