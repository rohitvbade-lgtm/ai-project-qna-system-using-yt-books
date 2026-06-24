from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from app.agents.book_rag_agent import run_book_rag_agent
from app.agents.judge_agent import judge_agent_answer
from app.agents.synthesis_agent import synthesize_final_answer
from app.agents.youtube_agent import run_youtube_agent
from app.config import get_settings
from app.graph.prompts import SUPERVISOR_ROUTING_PROMPT
from app.graph.state import JudgeResult, GeneralResearchState
from app.llm_client import MissingLLMConfigurationError, generate_text
from app.runtime_logging import get_logger

logger = get_logger(__name__)


class SupervisorRouteDecision(BaseModel):
    route_decision: str
    route_reason: str


def _heuristic_choose_route(question: str) -> tuple[str, str]:
    lowered = question.lower()
    mentions_video = any(token in lowered for token in ("youtube", "video", "videos", "transcript"))
    mentions_library = any(
        token in lowered
        for token in ("uploaded", "book", "books", "document", "documents", "pdf", "library")
    )
    asks_for_both = "both" in lowered or any(
        token in lowered for token in ("compare", "contrast", "versus", " vs ")
    )

    if asks_for_both or (mentions_video and mentions_library):
        return "both", "The user explicitly asked for cross-source comparison."
    if mentions_video:
        return "youtube", "The user asked for video-based or transcript-based evidence."
    if mentions_library:
        return "books", "The user explicitly referenced uploaded source material."
    return "books", "Defaulting to the local library for a grounded knowledge answer."


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _normalize_route_decision(route_decision: str) -> str:
    normalized = route_decision.strip().lower()
    if normalized in {"book", "books", "library", "rag"}:
        return "books"
    if normalized in {"youtube", "video", "videos"}:
        return "youtube"
    if normalized in {"both", "compare", "comparison"}:
        return "both"
    raise ValueError(f"Unsupported route decision returned by supervisor LLM: {route_decision}")


def _llm_choose_route(question: str) -> tuple[str, str]:
    response_text = generate_text(
        system_prompt=SUPERVISOR_ROUTING_PROMPT,
        user_prompt=(
            f"Question: {question}\n\n"
            "Return JSON only. Select the single best route_decision "
            "and give a concise route_reason."
        ),
    )
    payload = json.loads(_strip_json_fence(response_text))
    decision = SupervisorRouteDecision.model_validate(payload)
    route_decision = _normalize_route_decision(decision.route_decision)
    route_reason = decision.route_reason.strip()
    if not route_reason:
        raise ValueError("Supervisor LLM returned an empty route_reason.")
    return route_decision, route_reason


def _choose_route(question: str) -> tuple[str, str, list[str]]:
    settings = get_settings()
    if not settings.llm_enabled:
        route_decision, route_reason = _heuristic_choose_route(question)
        return route_decision, route_reason, []

    try:
        route_decision, route_reason = _llm_choose_route(question)
        return route_decision, route_reason, []
    except (
        MissingLLMConfigurationError,
        ValidationError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        logger.warning("router: LLM routing failed, using heuristic fallback: %s", exc)
        route_decision, route_reason = _heuristic_choose_route(question)
        return (
            route_decision,
            f"{route_reason} Supervisor LLM routing fallback applied.",
            [f"Supervisor LLM routing fallback: {exc}"],
        )


def supervisor_router_node(state: GeneralResearchState) -> GeneralResearchState:
    route_decision, route_reason, route_errors = _choose_route(state["user_question"])
    logger.info("router: selected route=%s (%s)", route_decision, route_reason)
    return {
        **state,
        "route_decision": route_decision,
        "route_reason": route_reason,
        "retry_count": state.get("retry_count", 0),
        "errors": [*state.get("errors", []), *route_errors],
    }


def book_agent_node(state: GeneralResearchState) -> GeneralResearchState:
    retry_instruction = None
    if state.get("retry_target") == "books" and state.get("book_judgement"):
        retry_instruction = state["book_judgement"].retry_instruction
    logger.info(
        "books: starting book agent%s",
        " with retry guidance" if retry_instruction else "",
    )
    answer = run_book_rag_agent(state["user_question"], retry_instruction=retry_instruction)
    logger.info("books: book agent finished with %s citations", len(answer.citations))
    return {**state, "book_answer": answer, "retry_target": None}


def youtube_agent_node(state: GeneralResearchState) -> GeneralResearchState:
    retry_instruction = None
    if state.get("retry_target") == "youtube" and state.get("youtube_judgement"):
        retry_instruction = state["youtube_judgement"].retry_instruction
    logger.info(
        "youtube: starting YouTube agent%s",
        " with retry guidance" if retry_instruction else "",
    )
    answer = run_youtube_agent(state["user_question"], retry_instruction=retry_instruction)
    logger.info("youtube: YouTube agent finished with %s citations", len(answer.citations))
    return {**state, "youtube_answer": answer, "retry_target": None}


def judge_node(state: GeneralResearchState) -> GeneralResearchState:
    logger.info("judge: evaluating agent responses")
    updated_state = dict(state)
    if state.get("book_answer"):
        updated_state["book_judgement"] = judge_agent_answer(
            state["user_question"], state["book_answer"]
        )
        logger.info(
            "judge: book answer pass=%s risk=%s",
            updated_state["book_judgement"].passed,
            updated_state["book_judgement"].hallucination_risk,
        )
    if state.get("youtube_answer"):
        updated_state["youtube_judgement"] = judge_agent_answer(
            state["user_question"],
            state["youtube_answer"],
        )
        logger.info(
            "judge: YouTube answer pass=%s risk=%s",
            updated_state["youtube_judgement"].passed,
            updated_state["youtube_judgement"].hallucination_risk,
        )
    return updated_state


def _pick_retry_target(state: GeneralResearchState) -> str | None:
    candidates: list[tuple[str, JudgeResult]] = []
    if state.get("book_judgement") and not state["book_judgement"].passed:
        candidates.append(("books", state["book_judgement"]))
    if state.get("youtube_judgement") and not state["youtube_judgement"].passed:
        candidates.append(("youtube", state["youtube_judgement"]))

    if not candidates:
        return None

    def _score(result: JudgeResult) -> float:
        return (
            result.relevance_to_question
            + result.source_support
            + result.citation_quality
            + result.completeness
        )

    return min(candidates, key=lambda item: _score(item[1]))[0]


def retry_router_node(state: GeneralResearchState) -> GeneralResearchState:
    target = _pick_retry_target(state)
    if target is None:
        logger.info("retry: no retry target selected")
        return {**state, "retry_target": None}
    logger.info(
        "retry: retrying %s agent (attempt %s)",
        target,
        state.get("retry_count", 0) + 1,
    )
    return {
        **state,
        "retry_count": state.get("retry_count", 0) + 1,
        "retry_target": target,
    }


def synthesis_node(state: GeneralResearchState) -> GeneralResearchState:
    logger.info("synthesis: combining accepted evidence into final answer")
    final_answer = synthesize_final_answer(
        question=state["user_question"],
        book_answer=state.get("book_answer"),
        youtube_answer=state.get("youtube_answer"),
        book_judgement=state.get("book_judgement"),
        youtube_judgement=state.get("youtube_judgement"),
    )
    logger.info("synthesis: final answer ready")
    return {**state, "final_answer": final_answer}


def route_after_supervisor(state: GeneralResearchState) -> str:
    if state["route_decision"] == "books":
        return "book_agent_node"
    if state["route_decision"] == "youtube":
        return "youtube_agent_node"
    return "book_agent_node"


def route_after_book(state: GeneralResearchState) -> str:
    if state.get("route_decision") == "both" and not state.get("youtube_answer"):
        return "youtube_agent_node"
    return "judge_node"


def route_after_youtube(state: GeneralResearchState) -> str:
    return "judge_node"


def route_after_judge(state: GeneralResearchState) -> str:
    settings = get_settings()
    retry_target = _pick_retry_target(state)
    if retry_target and state.get("retry_count", 0) < settings.max_agent_retries:
        return "retry_router_node"
    return "synthesis_node"


def route_after_retry(state: GeneralResearchState) -> str:
    if state.get("retry_target") == "books":
        return "book_agent_node"
    return "youtube_agent_node"
