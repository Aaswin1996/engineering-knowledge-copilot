import math

from app.rag_service import is_relevant


def test_relevant_chunk_is_accepted(monkeypatch):
    monkeypatch.setenv("RAG_MAX_COSINE_DISTANCE", "0.35")
    assert is_relevant([{"distance": 0.20}]) is True


def test_irrelevant_chunk_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_MAX_COSINE_DISTANCE", "0.35")
    assert is_relevant([{"distance": 0.60}]) is False


def test_empty_results_are_rejected():
    assert is_relevant([]) is False


def test_nan_distance_is_rejected():
    assert is_relevant([{"distance": float("nan")}]) is False


def test_infinite_distance_is_rejected():
    assert is_relevant([{"distance": math.inf}]) is False


def test_invalid_distance_is_rejected():
    assert is_relevant([{"distance": "not-a-number"}]) is False
