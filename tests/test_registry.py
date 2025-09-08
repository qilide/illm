import os
import pytest

from illm.model.core import provider_registry
from illm.model.providers import register_default_providers


def test_registry_register_and_available():
    # 注册一次（幂等）
    register_default_providers(provider_registry)
    av = provider_registry.available()
    assert "transformers" in av
    assert "ollama" in av
    assert "tgi" in av
    assert "vllm" in av
    assert "sglang" in av


def test_registry_create_transformers_default():
    cfg = {
        "framework": "transformers",
        "model": "sshleifer/tiny-gpt2",
        "embedding_model": "sshleifer/tiny-distilbert-base-cased",
        "device": "cpu",
        "max_tokens": 8,
    }
    register_default_providers(provider_registry)
    provider = provider_registry.create("transformers", cfg)
    caps = provider.capabilities
    assert caps.supports_text_generation is True
    assert caps.supports_embeddings is True
    assert caps.supports_vision is False