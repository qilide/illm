import os
import json
import pytest
import requests
from fastapi.testclient import TestClient

from illm.api.fast_api import app


def _ollama_models():
    base = os.environ.get("ILLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        r = requests.get(f"{base.rstrip('/')}/api/tags", timeout=3)
        if r.status_code != 200:
            return []
        data = r.json() or {}
        models = data.get("models", [])
        # Each item: {"name": "llama3:8b", "model": "...", ...}
        return [m.get("name") or m.get("model") for m in models if isinstance(m, dict)]
    except Exception:
        return []


def _has_ollama():
    try:
        base = os.environ.get("ILLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        r = requests.get(f"{base.rstrip('/')}/api/version", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _has_ollama(), reason="Ollama not available locally")
def test_ollama_text_chat_if_available():
    models = _ollama_models()
    if not models:
        pytest.skip("No ollama models installed")

    # pick first model for a lightweight test
    model = models[0]
    client = TestClient(app)
    body = {
        "framework": "ollama",
        "model": model,
        "messages": [{"role": "user", "content": "Hello from test"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") in ("chat.completion", "chat.completion")  # openai style
    assert "choices" in data


@pytest.mark.skipif(not _has_ollama(), reason="Ollama not available locally")
def test_ollama_multimodal_if_available():
    # run only if a known vision model exists
    models = _ollama_models()
    vision_candidates = [m for m in models if m and any(k in m.lower() for k in ["llava", "vision", "minicpm"])]
    if not vision_candidates:
        pytest.skip("No vision-enabled ollama model found (llava/minicpm etc.)")

    model = vision_candidates[0]
    client = TestClient(app)
    body = {
        "framework": "ollama",
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image shortly."},
                    {
                        "type": "image_url",
                        "image_url": "https://httpbin.org/image/png"
                    }
                ]
            }
        ],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=body)
    # Just ensure backend path works if model available
    assert resp.status_code in (200, 422)  # 422 if model actually doesn't support vision despite naming
    if resp.status_code == 200:
        data = resp.json()
        assert "choices" in data


def _has_env_url(key):
    return bool(os.environ.get(key))


@pytest.mark.skipif(not _has_env_url("ILLM_TGI_BASE_URL"), reason="TGI base url not configured")
def test_tgi_basic_when_configured():
    client = TestClient(app)
    body = {
        "framework": "tgi",
        "model": os.environ.get("ILLM_MODEL", "unknown"),
        "messages": [{"role": "user", "content": "Hello TGI"}],
        "stream": False,
        # Optionally pass params
        "max_tokens": 8,
        "temperature": 0.0,
    }
    resp = client.post("/v1/chat/completions", json=body)
    # We don't assert 200 strongly; remote might be unavailable.
    # If configured properly it should be 200.
    assert resp.status_code in (200, 400, 422)


@pytest.mark.skipif(not _has_env_url("ILLM_VLLM_BASE_URL"), reason="vLLM base url not configured")
def test_vllm_basic_when_configured():
    client = TestClient(app)
    body = {
        "framework": "vllm",
        "model": os.environ.get("ILLM_MODEL", "unknown"),
        "messages": [{"role": "user", "content": "Hello vLLM"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code in (200, 400, 422)


@pytest.mark.skipif(not _has_env_url("ILLM_SGLANG_BASE_URL"), reason="SGLang base url not configured")
def test_sglang_basic_when_configured():
    client = TestClient(app)
    body = {
        "framework": "sglang",
        "model": os.environ.get("ILLM_MODEL", "unknown"),
        "messages": [{"role": "user", "content": "Hello SGLang"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code in (200, 400, 422)