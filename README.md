# Engineering Knowledge Copilot — RAG

A learning project that demonstrates a Retrieval-Augmented Generation (RAG) assistant for engineering knowledge such as SOPs, Architecture Decision Records (ADRs), runbooks, and troubleshooting guides.

The goal is to let an engineer ask a question in natural language, retrieve relevant passages from private documentation, and receive a response grounded in those passages with source references.

> This repository is a local learning/reference implementation. It is not a production-ready corporate knowledge service. The relevance threshold is a heuristic, and citations are checked against retrieved document IDs but are not independently proven to entail every generated claim.

## Architecture

### Ingestion (offline / separate pipeline)

1. Extract text from source documentation.
2. Split documents into chunks while retaining document IDs, titles, and section metadata.
3. Generate embeddings with Gemini `gemini-embedding-001`.
4. Store chunk text, metadata, and 3,072-dimensional vectors in PostgreSQL with pgvector.

The included `engineering_knowledge_copilot_rag_reference.ipynb` notebook documents the local ingestion and retrieval workflow.

### Query (online)

1. `POST /ask` validates the user's question.
2. Gemini embeds the question.
3. PostgreSQL/pgvector ranks chunks using cosine distance (`<=>`). Lower distance means a closer vector match; it is not a confidence percentage.
4. A configurable threshold can abstain when the nearest result is too far away.
5. Gemini generates an answer from retrieved context and returns the document IDs it cites.
6. The API filters citations to IDs present in the retrieved results and removes duplicates.

```text
Internal docs -> extract/chunk -> Gemini embeddings -> PostgreSQL + pgvector
                                                        ^
                                                        |
Question -> query embedding -> similarity search -> relevant chunks
                                                   -> Gemini grounded answer
                                                   -> answer + source metadata
```

## Stack

- Python 3.12+
- FastAPI + Uvicorn
- Gemini API (`google-genai`) for embeddings and generation
- PostgreSQL + pgvector
- pytest + FastAPI TestClient for automated tests

## Setup

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` locally and set your own `GEMINI_API_KEY`. Do not commit `.env` or share the key. If a key has been accidentally shared, revoke/rotate it in the provider console.

### 3. Start PostgreSQL with pgvector

Docker Desktop must be running.

```bash
docker compose up -d postgres
```

The schema is initialized on the first database volume creation. If you already have a database/table, ensure `document_chunks` has the columns in `sql/schema.sql` and `embedding VECTOR(3072)`. The SQL file creates the extension, table, and indexes for a new database.

For a fresh database, use the connection string shown in `.env.example`. Change the development credentials before using this outside a local machine.

### 4. Ingest sample documents

Open `engineering_knowledge_copilot_rag_reference.ipynb` and run its setup, chunking, embedding, and upsert cells. The notebook uses sample engineering documents and your local `.env` configuration.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

- Health: <http://127.0.0.1:8000/health>
- Swagger: <http://127.0.0.1:8000/docs>
- Ask: `POST /ask`

Example request:

```json
{
  "question": "What happens if PostgreSQL fails during order creation?"
}
```

Example response shape:

```json
{
  "answer": "The request must fail safely [RUNBOOK-007].",
  "sources": [
    {
      "document_id": "RUNBOOK-007",
      "title": "Order Service Database Failure",
      "section": "Failure Handling"
    }
  ]
}
```

The exact answer depends on the model response and indexed sample data. `POST /debug/retrieve` is a development endpoint that exposes retrieved chunk text and similarity distances; do not expose it publicly without appropriate authentication and access controls.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API credential | Required |
| `PG_DSN` | PostgreSQL connection string | Required |
| `GEMINI_EMBEDDING_MODEL` | Query/document embedding model | `gemini-embedding-001` |
| `GEMINI_GENERATION_MODEL` | Answer-generation model | `gemini-3.5-flash-lite` |
| `RAG_MAX_COSINE_DISTANCE` | Initial maximum nearest-neighbor distance | `0.35` |

The generation model must be available to your API key; set it to a supported model in your environment. The distance threshold is dataset-specific and must be calibrated using labeled answerable and unanswerable questions.

## Tests

Tests mock external retrieval and generation dependencies, so they do not require a running PostgreSQL server or Gemini API key.

```bash
python -m pytest -v
```

Coverage includes relevance-threshold behavior, non-finite distances, request validation, abstention, source filtering/deduplication, and dependency failures.


## Limitations and next steps

- Calibrate retrieval thresholds with a representative evaluation set (Recall@K, precision, abstention accuracy).
- Evaluate citation correctness against supporting chunks, not only document-ID membership.
- Add incremental ingestion, deletion/version handling, and document freshness tracking.
- Add authentication and document-level authorization before indexing sensitive internal documentation.
- Add rate limits, timeouts/retries, connection pooling, structured logs, tracing, and usage/cost monitoring.
- Consider hybrid lexical + vector retrieval and reranking when evaluation demonstrates a need.
- Add CI to run linting, type checking, and tests on each change.
