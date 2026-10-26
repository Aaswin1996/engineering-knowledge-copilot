from fastapi.testclient import TestClient

from app import main
from app.main import ABSTENTION_ANSWER, app

client = TestClient(app)


def make_chunk(document_id="RUNBOOK-007", distance=0.20, content="Fail safely."):
    return {
        "document_id": document_id,
        "title": "Order Service Database Failure",
        "section": "Failure Handling",
        "content": content,
        "distance": distance,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_only_cited_sources(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve_chunks",
        lambda question: [
            make_chunk(),
            make_chunk(content="Retry transient failures."),
            make_chunk(document_id="ADR-014", content="PostgreSQL is the source of truth."),
        ],
    )
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, chunks: {
            "answer": "The request must fail safely [RUNBOOK-007].",
            "cited_document_ids": ["RUNBOOK-007", "NOT-RETRIEVED"],
        },
    )

    response = client.post("/ask", json={"question": "What happens if PostgreSQL fails?"})

    assert response.status_code == 200
    body = response.json()
    assert "fail safely" in body["answer"]
    assert [source["document_id"] for source in body["sources"]] == ["RUNBOOK-007"]


def test_ask_deduplicates_cited_sources(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieve_chunks",
        lambda question: [make_chunk(), make_chunk(content="Retry safely.")],
    )
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, chunks: {
            "answer": "Fail safely [RUNBOOK-007].",
            "cited_document_ids": ["RUNBOOK-007", "RUNBOOK-007"],
        },
    )

    response = client.post("/ask", json={"question": "What happens if PostgreSQL fails?"})

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1


def test_ask_abstains_when_evidence_is_irrelevant(monkeypatch):
    monkeypatch.setattr(main, "retrieve_chunks", lambda question: [make_chunk(distance=0.80)])

    response = client.post("/ask", json={"question": "What is the capital of Japan?"})

    assert response.status_code == 200
    assert response.json() == {"answer": ABSTENTION_ANSWER, "sources": []}


def test_ask_rejects_too_short_question():
    response = client.post("/ask", json={"question": "Hi"})
    assert response.status_code == 422


def test_ask_handles_retrieval_failure(monkeypatch):
    def fail(_question):
        raise RuntimeError("internal configuration detail")

    monkeypatch.setattr(main, "retrieve_chunks", fail)
    response = client.post("/ask", json={"question": "What happens if PostgreSQL fails?"})

    assert response.status_code == 503
    assert "internal configuration detail" not in response.text
