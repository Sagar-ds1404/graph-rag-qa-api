# College Viva Notes & Oral Exam Guide
## Graph RAG Based Multi-Document Question Answering API

---

## ⚡ 60-Second Project Explanation (Viva Elevator Pitch)

> *"Good morning / afternoon professors. My project is a **Graph RAG Multi-Document Question Answering API** built using Python, FastAPI, FAISS, Neo4j, and Google Gemini.*
>
> *Traditional RAG systems only use vector search. While vector search is good at finding semantically similar paragraphs, it struggles to connect related concepts scattered across multiple documents or understand structural relationships.*
>
> *To solve this, my project implements **Hybrid Graph RAG**:*
> 1. *When PDF documents are uploaded, text is extracted and split into chunks.*
> 2. *We compute dense embeddings using `all-MiniLM-L6-v2` and store them in a **FAISS** vector index.*
> 3. *Simultaneously, we extract technical entities and relationships, storing them as a structured **Knowledge Graph in Neo4j**.*
> 4. *When a user asks a question, the API performs **Hybrid Retrieval**: it finds the most similar chunks using FAISS and discovers connected entities using Neo4j.*
> 5. *Finally, both contexts are passed to **Google Gemini** with a strictly grounded prompt, returning an accurate, hallucination-free answer with page and chunk citations.*
>
> *This produces higher factual accuracy and enables relational reasoning that standard RAG cannot achieve."*

---

## 📚 Core Viva Questions & Answers

### 1. What is an API?
An **API (Application Programming Interface)** is a set of defined rules, protocols, and endpoints that allows different software applications to communicate with each other. In our project, client applications (like a browser or frontend) communicate with our Python backend over HTTP.

### 2. What is REST?
**REST (Representational State Transfer)** is an architectural style for building networked web services. It uses standard HTTP methods:
- `GET`: Read / retrieve data (e.g. `GET /documents`, `GET /health`)
- `POST`: Create / submit data (e.g. `POST /documents/upload`, `POST /query`)
- It is **stateless**, meaning each request contains all the information needed to complete it.

### 3. What is FastAPI?
**FastAPI** is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.
**Key advantages**:
- Extremely fast (built on top of Starlette and ASGI server Uvicorn).
- Automatic data validation using Pydantic.
- Automatic interactive documentation (Swagger UI at `/docs` and ReDoc at `/redoc`).

### 4. What is Pydantic?
**Pydantic** is a data validation and parsing library for Python. It enforces type hints at runtime. If a client sends invalid data (e.g. an empty string or missing field), Pydantic automatically catches it and returns a clear `422 Unprocessable Entity` HTTP error with details on what was wrong.

### 5. What is RAG?
**RAG (Retrieval-Augmented Generation)** is a technique where an external knowledge base is searched to find relevant facts before sending a prompt to a Large Language Model.
- **Why it matters**: LLMs have cut-off dates and can hallucinate. RAG injects your private, real-time documents into the prompt so the LLM answers based solely on verified facts.

### 6. What is Graph RAG?
**Graph RAG** combines knowledge graphs with traditional RAG. Instead of treating text solely as flat bags of words or vectors, it extracts entities (concepts, tools, names) and the explicit relationships connecting them (e.g., `(:FastAPI)-[:BASED_ON]->(:Starlette)`). This enables multi-hop reasoning across multiple pages or documents.

### 7. What are Embeddings?
An **embedding** is a numerical representation of text as a dense vector of floating-point numbers (e.g., a list of 384 numbers). Words, sentences, or paragraphs with similar meanings are located close together in high-dimensional vector space.

### 8. Why `sentence-transformers` and `all-MiniLM-L6-v2`?
- **sentence-transformers** provides pre-trained models specifically tuned to produce high-quality semantic representations of entire sentences and paragraphs (unlike basic word embeddings).
- **`all-MiniLM-L6-v2`** is lightweight (only ~80MB), fast, outputs a 384-dimensional vector, and achieves top performance in semantic similarity benchmarks while running efficiently on CPU.

### 9. What is FAISS?
**FAISS (Facebook AI Similarity Search)** is an open-source library developed by Meta for high-speed similarity search and clustering of dense vectors.

### 10. Why FAISS?
Traditional SQL or relational databases are not designed to compute cosine similarity or Euclidean distance across millions of multi-dimensional vectors. FAISS is written in optimized C++ with Python bindings and can search thousands of vectors in microseconds.

### 11. What is Neo4j?
**Neo4j** is the world's leading native **Graph Database Management System**. It stores data as nodes (entities) and relationships (directed edges) with key-value properties.

### 12. What is a Knowledge Graph?
A **Knowledge Graph** is a structured representation of real-world knowledge. It represents facts as subject-predicate-object triples, for example:
- `(FastAPI) -[:USES]-> (Pydantic)`
- `(RAG) -[:REQUIRES]-> (Embeddings)`

### 13. What is Cypher?
**Cypher** is the declarative query language used by Neo4j (analogous to SQL for relational databases).
Example used in our project:
```cypher
MATCH (e:Entity)
WHERE toLower(e.name) = toLower($entity_name)
OPTIONAL MATCH (e)-[r:RELATED_TO]-(related:Entity)
RETURN e.name AS entity, collect(DISTINCT related.name) AS related_entities
```
We always use **parameterized queries** (e.g., `$entity_name`) to prevent Cypher injection vulnerabilities.

### 14. Why use BOTH Vector and Graph Retrieval?
- **Vector Retrieval**: Captures semantic nuance, synonyms, and descriptive passages even when exact terminology differs.
- **Graph Retrieval**: Captures precise relationships, hierarchies, and multi-hop connections that span across different documents.
- **Hybrid Retrieval**: Merges both context streams so the LLM has both rich descriptive text and exact relational facts.

### 15. What happens step-by-step when a user asks a question?
1. **Request Received**: `POST /query` validates the input JSON using Pydantic.
2. **Query Vectorization**: The question string is converted into a 384-dimensional vector via `all-MiniLM-L6-v2`.
3. **Vector Search**: FAISS calculates cosine similarity against indexed chunks and retrieves the top 5 relevant text passages.
4. **Graph Extraction**: Technical candidate entities in the question are parsed.
5. **Graph Traversal**: Neo4j executes parameterized Cypher to fetch neighboring entities and connected chunks.
6. **Context Aggregation**: Vector passages and graph relationships are merged, and citations are deduplicated.
7. **Prompt Construction**: A strict grounding prompt is composed.
8. **LLM Generation**: Google Gemini generates the answer based solely on the context.
9. **Response Formatted**: The JSON response is returned containing the answer and source citations (document name, chunk ID, page number).

### 16. What does Gemini do?
Google Gemini acts as the **reasoning and generation engine**. It reads the retrieved context, verifies whether the context answers the user's question, structures the response, and ensures no ungrounded claims are made.

### 17. What are the limitations of this MVP?
- In-memory / local FAISS index (suitable for single-server setups, not distributed clusters).
- Ingestion runs synchronously during the HTTP upload request (ideal for demos; for enterprise production, a background queue like Celery would be added).
- Simple heuristic entity extraction fallback if Gemini API is not reachable.

### 18. What future improvements can be made?
- Add a Cross-Encoder Reranker to re-score hybrid results before feeding them to Gemini.
- Implement an agentic query rewriter (HyDE or query expansion).
- Add support for DOCX, TXT, and Markdown files in addition to PDF.
- Deploy a distributed vector store (e.g. Qdrant or Milvus) and Neo4j cluster.

---

## 🎯 Viva Tips for High Marks
1. **Explain the Hybrid Concept First**: Emphasize that standard RAG only does vector search, but your project adds graph knowledge.
2. **Mention Safety**: Mention parameterized Cypher queries to prevent injection attacks and strict system prompts to prevent LLM hallucinations.
3. **Point out Modularity**: Show the clean separation in `app/` (`vector_store.py`, `graph_store.py`, `document_processor.py`, `llm.py`, `retrieval.py`).
4. **Demonstrate Swagger**: Open `http://127.0.0.1:8000/docs` in the browser to demonstrate the live interactive documentation.
