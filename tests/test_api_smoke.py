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
                b"The document project uses FastAPI and scikit-learn for retrieval.",
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
