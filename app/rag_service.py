"""Embedding, vector retrieval, relevance checks, and grounded generation."""

import json
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"


class RetrievalUnavailableError(RuntimeError):
    """Raised when the vector database or its client dependencies are unavailable."""


def is_relevant(chunks: list[dict]) -> bool:
    """Apply an initial cosine-distance threshold to the nearest chunk.

    This threshold is a heuristic and must be calibrated with a labeled
    evaluation set before production use. Lower cosine distance means closer.
    """
    if not chunks:
        return False

    try:
        best_distance = float(chunks[0]["distance"])
        max_distance = float(os.getenv("RAG_MAX_COSINE_DISTANCE", "0.35"))
    except (KeyError, TypeError, ValueError):
        logger.warning("Invalid distance or RAG threshold")
        return False

    if not math.isfinite(best_distance) or not math.isfinite(max_distance):
        return False

    return best_distance <= max_distance


def _get_client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured in the environment")
    return genai.Client(api_key=api_key)


def retrieve_chunks(question: str, top_k: int = 3) -> list[dict]:
    """Embed a question and retrieve its nearest chunks from pgvector."""
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN must be configured in the environment")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        import psycopg
        from pgvector import Vector
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise RetrievalUnavailableError(
            "Database dependencies are not installed; install requirements.txt"
        ) from exc

    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )
    if not result.embeddings or not result.embeddings[0].values:
        raise RuntimeError("Embedding service returned an empty embedding")
    query_embedding = result.embeddings[0].values

    try:
        with psycopg.connect(dsn) as conn:
            register_vector(conn)
            rows = conn.execute(
                """
                SELECT document_id, title, section, content,
                       embedding <=> %s AS distance
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(query_embedding), Vector(query_embedding), top_k),
            ).fetchall()
    except psycopg.Error as exc:
        raise RetrievalUnavailableError(
            "Could not connect to or query the vector database"
        ) from exc

    return [
        {
            "document_id": row[0],
            "title": row[1],
            "section": row[2],
            "content": row[3],
            "distance": float(row[4]),
        }
        for row in rows
    ]


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Generate a JSON answer and document IDs using retrieved context only."""
    if not chunks:
        return {"answer": "Insufficient evidence to answer.", "cited_document_ids": []}

    client = _get_client()
    context = "\n\n".join(
        f"[{chunk['document_id']} | {chunk['title']} | {chunk['section']}]\n"
        f"{chunk['content']}"
        for chunk in chunks
    )
    allowed_ids = list(dict.fromkeys(chunk["document_id"] for chunk in chunks))

    prompt = f"""
You are an engineering knowledge assistant.
Answer using only the supplied context. Do not use outside knowledge or invent facts.
If the context does not contain enough evidence, say that you cannot answer from the available documents.
Cite each supported claim in the answer using [DOCUMENT-ID].
Only cite document IDs present in this context.
Return only valid JSON with this exact shape:
{{"answer": "answer text with citations", "cited_document_ids": ["DOCUMENT-ID"]}}
The cited_document_ids array must contain only source IDs that support the answer.
If the answer is insufficient, return an empty cited_document_ids array.

Allowed document IDs: {json.dumps(allowed_ids)}

Context:
{context}

Question:
{question}
""".strip()

    response = client.models.generate_content(
        model=os.getenv("GEMINI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        contents=prompt,
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned malformed JSON") from exc

    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object")
    answer = result.get("answer")
    cited_ids = result.get("cited_document_ids")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Gemini response is missing a valid answer")
    if not isinstance(cited_ids, list) or not all(
        isinstance(item, str) for item in cited_ids
    ):
        raise ValueError("Gemini response contains invalid cited_document_ids")

    # The API also filters these IDs against retrieved sources before returning.
    return {"answer": answer.strip(), "cited_document_ids": cited_ids}
