from __future__ import annotations

from typing import Any

from app.config import get_settings


class MissingLLMConfigurationError(RuntimeError):
    """Raised when a real LLM is requested but not configured."""


def build_openai_compatible_client(*, api_key: str, base_url: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise MissingLLMConfigurationError(
            "The openai package is not installed. Install project dependencies first."
        ) from exc

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def generate_text(*, system_prompt: str | None, user_prompt: str) -> str:
    settings = get_settings()
    if not settings.llm_enabled or not settings.resolved_llm_api_key:
        raise MissingLLMConfigurationError(
            "No LLM provider is configured. Set LLM_PROVIDER and related settings first."
        )

    client = build_openai_compatible_client(
        api_key=settings.resolved_llm_api_key,
        base_url=settings.resolved_llm_base_url,
    )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()
