"""FastAPI entry point for the Engineering Knowledge Copilot."""

import logging
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_service import (
    RetrievalUnavailableError,
    generate_answer,
    is_relevant,
    retrieve_chunks,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Engineering Knowledge Copilot",
    description=(
        "A retrieval-augmented assistant for engineering documentation "
        "such as SOPs, ADRs, and runbooks."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=1000,
        description="Question about the indexed engineering documentation.",
    )


class Source(BaseModel):
    document_id: str
    title: str
    section: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


ABSTENTION_ANSWER = (
    "I couldn't find sufficiently relevant evidence in the engineering "
    "documents to answer this question."
)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Report that the API process is responding (not a DB readiness check)."""
    return {"status": "ok"}


@app.get("/", tags=["operations"])
def root() -> dict[str, str]:
    return {"message": "Engineering Knowledge Copilot API", "docs": "/docs"}


@app.post("/debug/retrieve", tags=["debug"])
def debug_retrieve(request: AskRequest) -> dict:
    """Return retrieved chunks and distances; intended for local development."""
    try:
        chunks = retrieve_chunks(request.question)
    except (RuntimeError, RetrievalUnavailableError):
        logger.exception("Retrieval dependency failed")
        raise HTTPException(
            status_code=503,
            detail="Retrieval service is not currently available.",
        ) from None
    except Exception:
        logger.exception("Unexpected retrieval failure")
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve engineering context.",
        ) from None

    return {"question": request.question, "results": chunks}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest) -> AskResponse:
    """Retrieve evidence and generate an answer grounded in that evidence."""
    try:
        chunks = retrieve_chunks(request.question)
    except (RuntimeError, RetrievalUnavailableError):
        logger.exception("Retrieval dependency failed")
        raise HTTPException(
            status_code=503,
            detail="Retrieval service is not currently available.",
        ) from None
    except Exception:
        logger.exception("Unexpected retrieval failure")
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve engineering context.",
        ) from None

    if not is_relevant(chunks):
        return AskResponse(answer=ABSTENTION_ANSWER, sources=[])

    try:
        result = generate_answer(request.question, chunks)
    except (RuntimeError, ValueError, KeyError):
        logger.exception("Answer generation returned an invalid result")
        raise HTTPException(
            status_code=502,
            detail="The answer service returned an invalid response.",
        ) from None
    except Exception:
        logger.exception("Answer generation failed")
        raise HTTPException(
            status_code=502,
            detail="The answer service is not currently available.",
        ) from None

    answer = result["answer"]
    cited_ids = result["cited_document_ids"]

    # Deduplicate retrieved documents, preserving their retrieval order.
    available_sources: OrderedDict[str, Source] = OrderedDict()
    for chunk in chunks:
        document_id = chunk["document_id"]
        if document_id not in available_sources:
            available_sources[document_id] = Source(
                document_id=document_id,
                title=chunk["title"],
                section=chunk["section"],
            )

    # Only return citations that refer to documents actually retrieved.
    sources = [
        available_sources[document_id]
        for document_id in dict.fromkeys(cited_ids)
        if document_id in available_sources
    ]

    return AskResponse(answer=answer, sources=sources)
