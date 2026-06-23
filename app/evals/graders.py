from __future__ import annotations

from typing import Any

from app.graph.state import MusicResearchState
from app.llm_client import MissingLLMConfigurationError, generate_text


def route_accuracy(expected_route: str, actual_route: str) -> float:
    return 1.0 if expected_route == actual_route else 0.0


def citation_presence(answer_text: str) -> float:
    return 1.0 if "Sources" in answer_text or "source" in answer_text.lower() else 0.0


def answer_completeness(answer_text: str) -> float:
    return min(1.0, len(answer_text.split()) / 160)


def groundedness_heuristic(state: MusicResearchState) -> float:
    citations = 0
    for key in ("book_answer", "youtube_answer"):
        answer = state.get(key)
        if answer:
            citations += len(answer.citations)
    return min(1.0, citations / 4)


def optional_llm_judge(answer_text: str) -> dict[str, Any]:
    try:
        response_text = generate_text(
            system_prompt=None,
            user_prompt=(
                "Score this answer from 0 to 1 for groundedness and completeness. "
                "Return only a number.\n\n" + answer_text
            ),
        )
    except MissingLLMConfigurationError:
        return {"enabled": False, "score": None}

    try:
        return {"enabled": True, "score": float(response_text.strip())}
    except ValueError:
        return {"enabled": True, "score": None}
