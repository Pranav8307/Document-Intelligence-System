import time

import pytest

from services.document_service import chunk_text
from services.embedding_service import EmbeddingService
from utils import cache


def test_chunk_text_size_and_overlap():
    text = "".join(str(i % 10) for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)
    # the last 50 chars of a chunk are the first 50 of the next one
    assert chunks[0][-50:] == chunks[1][:50]
    assert len(chunks) == 3


def test_chunk_text_ignores_blank_input():
    assert chunk_text("   \n  ") == []


def test_retrieval_ranks_relevant_chunk_first():
    service = EmbeddingService()
    chunks = [
        "Bananas are yellow fruit rich in potassium.",
        "The server uses FastAPI and PostgreSQL for the backend API.",
        "Mount Everest is the tallest mountain above sea level.",
    ]
    service.index_chunks("test-doc", chunks)
    results = service.retrieve_top_k("test-doc", "Which backend database does the server use?", top_k=1)
    texts = [text for text, score in results]
    scores = {text: score for text, score in results}
    assert chunks[1] in texts
    # the matched chunk has the highest score; neighbours added for context score 0.0
    assert scores[chunks[1]] == max(scores.values()) and scores[chunks[1]] > 0


def test_retrieval_unknown_document_raises():
    with pytest.raises(KeyError):
        EmbeddingService().retrieve_top_k("missing-doc", "anything")


def test_cache_expires_after_ttl():
    cache.cache_set("k", "v", ttl=1)
    assert cache.cache_get("k") == "v"
    time.sleep(1.1)
    assert cache.cache_get("k") is None
