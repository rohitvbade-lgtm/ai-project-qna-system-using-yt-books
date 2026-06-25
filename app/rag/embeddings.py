from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from app.config import get_settings
from app.llm_client import build_openai_compatible_client
from app.runtime_logging import get_logger

logger = get_logger(__name__)


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
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.client = build_openai_compatible_client(api_key=api_key, base_url=base_url)
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_end = batch_start + len(batch)
            logger.info(
                "embeddings: requesting batch %s-%s of %s from model=%s",
                batch_start,
                batch_end - 1,
                len(texts),
                self.model,
            )
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:
                raise RuntimeError(
                    "Embedding batch failed for items "
                    f"{batch_start}-{batch_end - 1} using model '{self.model}'. "
                    "Reduce EMBEDDING_BATCH_SIZE or inspect the embedding server logs."
                ) from exc

            batch_embeddings = [list(item.embedding) for item in response.data]
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding provider returned an unexpected number of embeddings for "
                    f"items {batch_start}-{batch_end - 1}: expected {len(batch)}, "
                    f"received {len(batch_embeddings)}."
                )
            embeddings.extend(batch_embeddings)

        return embeddings


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
            batch_size=settings.embedding_batch_size,
        )

    if allow_fake or settings.embedding_provider == "fake":
        return FakeEmbeddingProvider(settings.embedding_dimension)

    if require_real:
        raise MissingEmbeddingConfigurationError(
            "No embedding provider is configured. Set EMBEDDING_PROVIDER and related settings."
        )

    return None
