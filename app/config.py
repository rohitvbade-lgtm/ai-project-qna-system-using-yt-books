from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default=(
            "postgresql+psycopg://knowledge_assistant:"
            "knowledge_assistant@localhost:5432/knowledge_assistant"
        ),
        alias="DATABASE_URL",
    )

    llm_provider: Literal["openai", "ollama", "groq", "none"] = Field(
        default="none",
        alias="LLM_PROVIDER",
    )
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    embedding_provider: Literal["openai", "ollama", "fake"] = Field(
        default="fake",
        alias="EMBEDDING_PROVIDER",
    )
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_base_url: str | None = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_dimension: int = Field(default=768, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    youtube_api_key: str | None = Field(default=None, alias="YOUTUBE_API_KEY")

    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="knowledge-assistant", alias="LANGSMITH_PROJECT")

    max_agent_retries: int = Field(default=2, alias="MAX_AGENT_RETRIES")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and bool(self.resolved_llm_api_key)

    @property
    def embeddings_enabled(self) -> bool:
        return self.embedding_provider in {"openai", "ollama"} and bool(
            self.resolved_embedding_api_key
        )

    @property
    def resolved_llm_api_key(self) -> str | None:
        if self.llm_provider == "ollama":
            return self.llm_api_key or "ollama"
        return self.llm_api_key

    @property
    def resolved_llm_base_url(self) -> str | None:
        if self.llm_provider == "ollama":
            return self.llm_base_url or "http://localhost:11434/v1"
        if self.llm_provider == "groq":
            return self.llm_base_url or "https://api.groq.com/openai/v1"
        return self.llm_base_url

    @property
    def resolved_embedding_api_key(self) -> str | None:
        if self.embedding_provider == "ollama":
            return self.embedding_api_key or "ollama"
        return self.embedding_api_key

    @property
    def resolved_embedding_base_url(self) -> str | None:
        if self.embedding_provider == "ollama":
            return self.embedding_base_url or "http://localhost:11434/v1"
        return self.embedding_base_url

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_tracing and self.langsmith_api_key)

    def apply_runtime_environment(self) -> None:
        os.environ["LANGSMITH_TRACING"] = "true" if self.langsmith_tracing else "false"
        if self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        if self.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
