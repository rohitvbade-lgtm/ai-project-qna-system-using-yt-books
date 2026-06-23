from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from app.config import get_settings
from app.llm_client import build_openai_compatible_client


class MissingEmbeddingConfigurationError(RuntimeError):
    """Raised when real embeddings are required but credentials are missing."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dimensions:
            for byte in digest:
                values.append((byte / 255.0) * 2 - 1)
                if len(values) >= self.dimensions:
                    break
            digest = hashlib.sha256(digest).digest()

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self.model = model
        self.client = build_openai_compatible_client(api_key=api_key, base_url=base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]


def get_embedding_provider(
    *,
    allow_fake: bool = False,
    require_real: bool = False,
) -> EmbeddingProvider | None:
    settings = get_settings()
    if settings.embeddings_enabled:
        return OpenAICompatibleEmbeddingProvider(
            model=settings.embedding_model,
            api_key=settings.resolved_embedding_api_key or "",
            base_url=settings.resolved_embedding_base_url,
        )

    if allow_fake or settings.embedding_provider == "fake":
        return FakeEmbeddingProvider(settings.embedding_dimension)

    if require_real:
        raise MissingEmbeddingConfigurationError(
            "No embedding provider is configured. Set EMBEDDING_PROVIDER and related settings."
        )

    return None
