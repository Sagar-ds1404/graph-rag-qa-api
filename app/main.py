"""Main FastAPI application for Graph RAG Multi-Document Question Answering API."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.document_processor import chunk_document, extract_text_from_pdf
from app.embeddings import generate_embeddings
from app.graph_store import graph_store
from app.llm import extract_entities_and_relations, generate_grounded_answer, is_gemini_configured
from app.retrieval import retrieve_context
from app.schemas import (
    DocumentItem,
    EntityDetailsResponse,
    GraphStatusResponse,
    HealthResponse,
    QuestionRequest,
    QuestionResponse,
    RootResponse,
    Source,
    UploadResponse,
)
from app.vector_store import vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for cross-origin accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["General"],
    summary="API Root Information",
    description="Returns service metadata, version, and running status.",
)
def root() -> RootResponse:
    """Return API basic information."""
    return RootResponse(
        message=settings.PROJECT_NAME,
        version=settings.VERSION,
        status="running",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
    description="Check the connectivity and readiness of Vector Store, Neo4j, and Gemini.",
)
def health_check() -> HealthResponse:
    """Return operational health status for core subsystems."""
    vector_status = "ready" if vector_store.get_total_chunks() > 0 else "empty"
    neo4j_status = "connected" if graph_store.is_available() else "unavailable"
    gemini_status = "configured" if is_gemini_configured() else "unavailable"

    overall_status = "healthy"
    return HealthResponse(
        status=overall_status,
        vector_store=vector_status,
        neo4j=neo4j_status,
        gemini=gemini_status,
    )


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    summary="Upload & Ingest PDF Documents",
    description="Accepts multiple PDF files, extracts text, chunks, embeds in FAISS, and builds graph in Neo4j.",
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF documents to ingest into the Graph RAG system.")
) -> UploadResponse:
    """Process and index one or more uploaded PDF documents."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided for upload.",
        )

    # Validate file formats first
    for file in files:
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: '{filename}'. Only PDF documents (.pdf) are supported.",
            )

    processed_count = 0
    total_chunks_created = 0

    for file in files:
        filename = file.filename or "document.pdf"
        doc_id = str(uuid.uuid4())[:8]
        saved_path = settings.DOCUMENTS_DIR / f"{doc_id}_{filename}"

        try:
            # 1. Save PDF file to storage
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 2. Extract text with page preservation
            pages_data = extract_text_from_pdf(saved_path)
            if not pages_data:
                logger.warning(f"No readable text extracted from '{filename}'.")
                continue

            # 3. Create overlapping chunks
            chunks = chunk_document(
                document_id=doc_id,
                document_name=filename,
                pages_data=pages_data,
                chunk_size_words=settings.CHUNK_SIZE_WORDS,
                overlap_words=settings.CHUNK_OVERLAP_WORDS,
            )
            if not chunks:
                logger.warning(f"No valid chunks generated for '{filename}'.")
                continue

            # 4. Generate embeddings and add to FAISS
            chunk_texts = [c["text"] for c in chunks]
            embeddings = generate_embeddings(chunk_texts)
            vector_store.add_chunks(chunks, embeddings)

            # 5. Extract knowledge graph entities & relationships and insert into Neo4j
            graph_store.add_document_and_chunks(doc_id, filename, chunks)

            # Extract entities per chunk (with Gemini or fallback)
            for c in chunks:
                extraction = extract_entities_and_relations(c["text"])
                entities = extraction.get("entities", [])
                relationships = extraction.get("relationships", [])
                if entities or relationships:
                    graph_store.add_entities_and_relations(
                        chunk_id=c["chunk_id"],
                        entities=entities,
                        relationships=relationships,
                    )

            processed_count += 1
            total_chunks_created += len(chunks)
            logger.info(f"Successfully processed '{filename}': {len(chunks)} chunks.")

        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}")
            # Individual document processing errors are handled gracefully without exposing stack traces
            continue
        finally:
            await file.close()

    if processed_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract readable content from the uploaded PDF document(s).",
        )

    return UploadResponse(
        status="success",
        documents_processed=processed_count,
        chunks_created=total_chunks_created,
        message=f"Successfully indexed {processed_count} document(s) and {total_chunks_created} chunk(s).",
    )


@app.get(
    "/documents",
    response_model=List[DocumentItem],
    tags=["Documents"],
    summary="List Ingested Documents",
    description="Returns metadata of all ingested documents along with chunk counts.",
)
def list_documents() -> List[DocumentItem]:
    """List summary information for all indexed documents."""
    docs = vector_store.get_documents_summary()
    return [
        DocumentItem(
            document_id=d["document_id"],
            document_name=d["document_name"],
            chunk_count=d["chunk_count"],
        )
        for d in docs
    ]


@app.post(
    "/query",
    response_model=QuestionResponse,
    tags=["RAG"],
    summary="Query Documents using Hybrid Graph RAG",
    description="Combines FAISS vector search and Neo4j graph retrieval to synthesize a grounded answer using Gemini.",
)
def query_documents(request: QuestionRequest) -> QuestionResponse:
    """Execute Hybrid Graph RAG query pipeline."""
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    if not is_gemini_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Gemini LLM is not configured. Please supply a valid GEMINI_API_KEY in the environment or .env file."
            ),
        )

    # 1. Perform Hybrid Retrieval (Vector + Knowledge Graph)
    retrieval_data = retrieve_context(question, top_k=settings.TOP_K_RESULTS)
    vector_context = retrieval_data.get("vector_context", [])
    graph_context = retrieval_data.get("graph_context", [])
    sources_data = retrieval_data.get("sources", [])

    # Format sources for response schema
    sources = [
        Source(
            document=s.get("document", "Unknown"),
            chunk_id=s.get("chunk_id", ""),
            page=int(s.get("page", 1)),
        )
        for s in sources_data
    ]

    # 2. Synthesize Grounded Answer with Gemini
    try:
        answer = generate_grounded_answer(
            question=question,
            vector_context=vector_context,
            graph_context=graph_context,
        )
    except Exception as e:
        logger.error(f"Error during answer generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating the answer: {str(e)}",
        )

    return QuestionResponse(
        question=question,
        answer=answer,
        sources=sources,
    )


@app.get(
    "/graph/status",
    response_model=GraphStatusResponse,
    tags=["Knowledge Graph"],
    summary="Check Neo4j Knowledge Graph Status",
    description="Returns node counts (Documents, Chunks, Entities) and connectivity status.",
)
def get_graph_status() -> GraphStatusResponse:
    """Retrieve graph database health and metrics."""
    if not graph_store.is_available():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "entity_count": None,
                "document_count": None,
                "chunk_count": None,
                "message": "Neo4j is not configured or reachable.",
            },
        )

    status_data = graph_store.get_status()
    return GraphStatusResponse(**status_data)


@app.get(
    "/graph/entities/{entity_name}",
    response_model=EntityDetailsResponse,
    tags=["Knowledge Graph"],
    summary="Lookup Entity in Knowledge Graph",
    description="Retrieve directly connected entities for a given entity name using safe parameterized Cypher.",
)
def lookup_entity(entity_name: str) -> EntityDetailsResponse:
    """Lookup an entity and its relationships in the Neo4j graph."""
    if not graph_store.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j graph database is not configured or reachable.",
        )

    result = graph_store.get_entity_and_related(entity_name)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity '{entity_name}' was not found in the knowledge graph.",
        )

    return EntityDetailsResponse(
        entity=result["entity"],
        related_entities=result["related_entities"],
    )
