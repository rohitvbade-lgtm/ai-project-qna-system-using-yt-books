from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag import embeddings as embedding_module


class _FakeEmbeddingsAPI:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def create(self, *, model: str, input: list[str]):
        self.calls.append(list(input))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError(f"simulated failure for model {model}")
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(index)]) for index, _ in enumerate(input)]
        )


def test_openai_compatible_provider_batches_embedding_requests(monkeypatch):
    api = _FakeEmbeddingsAPI()
    client = SimpleNamespace(embeddings=api)
    monkeypatch.setattr(
        embedding_module,
        "build_openai_compatible_client",
        lambda api_key, base_url=None: client,
    )

    provider = embedding_module.OpenAICompatibleEmbeddingProvider(
        model="test-model",
        api_key="test-key",
        batch_size=2,
    )

    embeddings = provider.embed_texts(["a", "b", "c", "d", "e"])

    assert api.calls == [["a", "b"], ["c", "d"], ["e"]]
    assert len(embeddings) == 5


def test_openai_compatible_provider_raises_with_batch_context(monkeypatch):
    api = _FakeEmbeddingsAPI(fail_on_call=2)
    client = SimpleNamespace(embeddings=api)
    monkeypatch.setattr(
        embedding_module,
        "build_openai_compatible_client",
        lambda api_key, base_url=None: client,
    )

    provider = embedding_module.OpenAICompatibleEmbeddingProvider(
        model="test-model",
        api_key="test-key",
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="Embedding batch failed for items 2-3"):
        provider.embed_texts(["a", "b", "c", "d"])
