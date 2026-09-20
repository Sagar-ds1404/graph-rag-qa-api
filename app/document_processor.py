"""Document processing: PDF text extraction, cleaning, and chunking."""

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List
import pypdf


def clean_text(text: str) -> str:
    """Clean extracted raw text from PDF.
    
    Removes redundant whitespace, non-printable characters,
    and normalizes line breaks while preserving sentence boundaries.
    """
    if not text:
        return ""
    # Remove control characters except tab and newline
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Normalize carriage returns
    text = re.sub(r"\r\n|\r", "\n", text)
    # Strip each line and normalize internal spaces
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(cleaned_lines)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Extract text from a PDF file page by page using pypdf.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        List of dictionaries with 'page' (1-indexed) and 'text'.
    """
    pages_data: List[Dict[str, Any]] = []
    reader = pypdf.PdfReader(str(file_path))
    
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if cleaned:
            pages_data.append({
                "page": page_num,
                "text": cleaned
            })
            
    return pages_data


def chunk_document(
    document_id: str,
    document_name: str,
    pages_data: List[Dict[str, Any]],
    chunk_size_words: int = 800,
    overlap_words: int = 100,
) -> List[Dict[str, Any]]:
    """Split extracted pages into meaningful, overlapping chunks.
    
    Preserves document_id, document_name, chunk_id, page, and text.
    Target chunk size: ~800 words, with ~100 words overlap.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    for page_item in pages_data:
        page_num = page_item["page"]
        text = page_item["text"]
        words = text.split()

        if not words:
            continue

        if len(words) <= chunk_size_words:
            # Page fits within one chunk
            chunk_text = " ".join(words)
            chunks.append({
                "document_id": document_id,
                "document_name": document_name,
                "chunk_id": f"{document_id}_c{chunk_counter}",
                "page": page_num,
                "text": chunk_text,
            })
            chunk_counter += 1
        else:
            # Slide a window across words with overlap
            step = max(1, chunk_size_words - overlap_words)
            for start in range(0, len(words), step):
                chunk_slice = words[start : start + chunk_size_words]
                if not chunk_slice:
                    continue
                chunk_text = " ".join(chunk_slice)
                chunks.append({
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_id": f"{document_id}_c{chunk_counter}",
                    "page": page_num,
                    "text": chunk_text,
                })
                chunk_counter += 1
                if start + chunk_size_words >= len(words):
                    break

    return chunks
