import os
import pytest
from fastapi.testclient import TestClient
from illm.api.fast_api import app


def test_healthz():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_chat_completions_transformers_non_stream():
    client = TestClient(app)
    body = {
        "framework": "transformers",
        "model": "sshleifer/tiny-gpt2",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") == "chat.completion"
    assert "choices" in data
    assert "model" in data


def test_embeddings_transformers():
    client = TestClient(app)
    body = {
        "framework": "transformers",
        "model": "sshleifer/tiny-gpt2",
        "embedding_model": "sshleifer/tiny-distilbert-base-cased",
        "input": ["hello", "world"]
    }
    resp = client.post("/v1/embeddings", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert "model" in data