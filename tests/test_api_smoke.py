import sys
import types

import numpy as np


class FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True):
        return np.array(
            [[float(len(text) or 1), 1.0] for text in texts], dtype="float32"
        )


fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = fake_sentence_transformers

from fastapi.testclient import TestClient

from main import app


def test_upload_and_ask_with_mock_llm(monkeypatch):
    monkeypatch.setenv("USE_MOCK_LLM", "true")
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    upload = client.post(
        "/documents/upload",
        files={
            "file": (
                "project.txt",
                b"The document project uses FastAPI and FAISS for retrieval.",
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    document_id = upload.json()["data"]["document_id"]

    answer = client.post(
        "/qa/ask",
        json={
            "document_id": document_id,
            "question": "Which framework does the project use?",
        },
    )
    assert answer.status_code == 200
    assert answer.json()["data"]["cached"] is False
    assert "[MOCK ANSWER]" in answer.json()["data"]["answer"]
