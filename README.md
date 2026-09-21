# Graph RAG Based Multi-Document Question Answering API

A production-grade, modular FastAPI application that implements **Hybrid Graph RAG (Retrieval-Augmented Generation)** across multiple PDF documents. By combining **FAISS dense vector similarity search** with a **Neo4j Knowledge Graph**, this system retrieves both semantically similar text passages and explicit entity relationship networks to provide grounded, hallucination-free answers using **Google Gemini**.

---

## 1. Problem Statement

Standard Large Language Models (LLMs) suffer from:
- **Hallucinations**: Generating confident but factually incorrect assertions.
- **Outdated Knowledge**: Inability to reason over private or newly published enterprise documents.
- **Loss of Structured Context in Pure Vector RAG**: Traditional Vector RAG retrieves isolated chunks based solely on semantic keyword/vector proximity, missing multi-hop structural connections between entities (e.g. *"Entity A is based on Entity B, which is used by System C"*).

---

## 2. Objective

The objective of this project is to build a high-performance, verifiable REST API that:
1. Ingests and processes multiple PDF documents.
2. Performs text extraction and sliding-window chunking while preserving exact page numbers and document IDs.
3. Computes vector embeddings and indexes them using **FAISS** for rapid nearest-neighbor lookup.
4. Extracts domain entities and typed relationships to build an associative knowledge graph in **Neo4j**.
5. Implements **Hybrid Retrieval**: simultaneously querying FAISS and Neo4j.
6. Synthesizes a strictly grounded answer with **Google Gemini**, citing source documents, chunks, and page numbers.

---

## 3. What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architectural pattern that supplements an LLM prompt with relevant external context retrieved from a custom knowledge base prior to generating an answer. Instead of retraining or fine-tuning weights, RAG dynamically provides ground-truth text chunks directly in the prompt context.

---

## 4. What is Graph RAG?

**Graph RAG** enhances standard RAG by integrating a structured **Knowledge Graph** into the retrieval loop.
- Documents are represented not just as isolated chunks of text, but as nodes (Documents, Chunks, Entities) and directed edges (`HAS_CHUNK`, `MENTIONS`, `RELATED_TO`).
- When a question is asked, graph traversal uncovers multi-hop relationships, topological entity connections, and conceptual hierarchies that pure vector similarity frequently overlooks.

---

## 5. Why Combine Vector + Graph Retrieval?

| Retrieval Mode | Primary Strength | Weakness |
| :--- | :--- | :--- |
| **Vector Retrieval (FAISS)** | Finds semantically similar passages even with varied phrasing. | Lacks relational understanding and multi-hop reasoning across distant sections. |
| **Graph Retrieval (Neo4j)** | Traverses explicit connections, dependencies, and entity relationships. | Sensitive to keyword variation and misses implicit, nuanced context. |
| **Hybrid Graph RAG (This API)** | **Best of both worlds**: Semantic context + structured relational graph. | Comprehensive, accurate, grounded answers with citations. |

---

## 6. Architecture

```text
                PDF DOCUMENTS
                      |
                      v
                TEXT EXTRACTION (pypdf)
                      |
                      v
             SLIDING-WINDOW CHUNKING
                  /        \
                 /          \
    EMBEDDINGS (MiniLM)   ENTITIES & RELATIONS
              |               |
              v               v
         FAISS INDEX    NEO4J GRAPH
              |               |
              \               /
               \             /
                HYBRID RETRIEVAL
                       |
                       v
               GEMINI LLM GROUNDING
                       |
                       v
                ANSWER + SOURCES
```

---

## 7. Technology Stack

- **Language**: Python 3.12+
- **Web Framework**: FastAPI, Uvicorn, Starlette
- **Data Validation**: Pydantic v2
- **PDF Extraction**: `pypdf`
- **Dense Vector Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Indexing**: `faiss-cpu`
- **Graph Database**: `neo4j` (Official Python Driver)
- **Generative AI / LLM**: `google-genai` (Official Google GenAI SDK)
- **Configuration**: `python-dotenv`
- **Testing**: `pytest`, `httpx`

---

## 8. Project Structure

```text
graph-rag-qa-api/
│
├── app/
│   ├── __init__.py               # Package marker
│   ├── main.py                   # FastAPI app, route definitions, error handlers
│   ├── config.py                 # Central settings & directory paths
│   ├── schemas.py                # Pydantic request & response schemas
│   ├── document_processor.py     # PDF text extraction, cleaning & word chunking
│   ├── embeddings.py             # Singleton sentence-transformers embedding loader
│   ├── vector_store.py           # FAISS index wrapper, search & metadata storage
│   ├── graph_store.py            # Neo4j driver with parameterized Cypher queries
│   ├── retrieval.py              # Hybrid retrieval orchestrator (Vector + Graph)
│   └── llm.py                    # Gemini client, entity extraction & grounded prompt
│
├── data/
│   ├── documents/                # Stored raw uploaded PDF documents
│   └── index/                    # Persisted FAISS index & metadata JSON
│
├── tests/
│   ├── __init__.py
│   └── test_api.py               # Complete test suite with mocked external services
│
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git exclusions (.env, .venv, indexes, logs)
├── requirements.txt              # Pinned/minimal dependencies
├── README.md                     # Comprehensive project documentation
├── VIVA_NOTES.md                 # College viva preparation & 60-second summary
└── run.py                        # Application entrypoint
```

---

## 9. Installation & Setup

### 9.1 Prerequisites
- Python 3.12 or higher installed.
- (Optional) A running Neo4j instance (local Desktop, Docker, or Neo4j AuraDB).
- (Optional) A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 9.2 Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 9.3 Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 10. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure the following variables in `.env`:

```ini
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash

# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
```

> **Note on External Services**:
> If either Neo4j or Gemini credentials are not yet supplied:
> - The application will **not** crash on startup.
> - Entity extraction safely falls back to local regex-based keyword extraction.
> - Vector search and health checks continue to work normally.
> - Query answering will return a descriptive HTTP 503 indicating that Gemini credentials must be configured.

---

## 11. Running the API

Start the development server using `run.py` or `uvicorn`:

```bash
python run.py
```

Or directly via Uvicorn:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will be available at:
- **API Base**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## 12. API Endpoints Table

| Method | Endpoint | Tag | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | General | Root API metadata and service status. |
| `GET` | `/health` | System | Health report for Vector Store, Neo4j, and Gemini. |
| `POST` | `/documents/upload` | Documents | Upload multiple PDF files for indexing. |
| `GET` | `/documents` | Documents | List all ingested documents and their chunk counts. |
| `POST` | `/query` | RAG | Query documents using Hybrid Graph RAG. |
| `GET` | `/graph/status` | Knowledge Graph | Check Neo4j node counts and connection status. |
| `GET` | `/graph/entities/{entity_name}` | Knowledge Graph | Retrieve an entity and its connected relationships. |

---

## 13. Usage Examples

### 13.1 Health Check
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Response**:
```json
{
  "status": "healthy",
  "vector_store": "ready",
  "neo4j": "connected",
  "gemini": "configured"
}
```

### 13.2 Upload PDF Documents
```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "files=@research_paper.pdf" \
  -F "files=@system_manual.pdf"
```
**Response**:
```json
{
  "status": "success",
  "documents_processed": 2,
  "chunks_created": 18,
  "message": "Successfully indexed 2 document(s) and 18 chunk(s)."
}
```

### 13.3 Ask a Question (Hybrid Graph RAG)
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the system handle hybrid retrieval and vector indexing?"}'
```
**Sample Response**:
```json
{
  "question": "How does the system handle hybrid retrieval and vector indexing?",
  "answer": "The system implements hybrid retrieval by executing concurrent queries across FAISS and Neo4j. In the vector phase, sentence embeddings are matched via Inner Product similarity. Simultaneously, candidate entities are extracted and resolved against Neo4j to retrieve relational neighborhoods. Both contexts are merged and provided to Google Gemini with a strict grounding prompt.",
  "sources": [
    {
      "document": "research_paper.pdf",
      "chunk_id": "8f3b12a0_c2",
      "page": 3
    },
    {
      "document": "system_manual.pdf",
      "chunk_id": "a9c043e1_c1",
      "page": 1
    }
  ]
}
```

### 13.4 Query Neo4j Entity Details
```bash
curl -X GET http://127.0.0.1:8000/graph/entities/FastAPI
```
**Response**:
```json
{
  "entity": "FastAPI",
  "related_entities": ["Python", "Starlette", "Uvicorn", "Pydantic"]
}
```

---

## 14. Testing

The project includes an automated test suite using `pytest` and FastAPI's `TestClient`. External dependencies (Gemini, Neo4j) are safely mocked so that tests run instantly and deterministically without network calls.

```bash
python -m pytest -v
```

---

## 15. Limitations & Future Work

### Current Limitations
- **Local FAISS Index**: FAISS runs in-process (`faiss-cpu`); for multi-replica horizontal scaling, a distributed vector database (e.g. Milvus, Qdrant) could be used.
- **Synchronous Ingestion**: Processing large 100+ page PDFs runs sequentially on the upload thread.

### Future Work
- **Async Task Queue**: Offload document parsing and graph building to Celery/Redis for enterprise scale.
- **Reranker Integration**: Add a Cross-Encoder reranker (e.g., `bge-reranker-large`) on top of the merged hybrid retrieval results.
- **Multi-Hop Graph Reasoning**: Implement dynamic LangGraph/agentic reasoning over Cypher queries for complex multi-step questions.
