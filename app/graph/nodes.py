from __future__ import annotations

from app.agents.book_rag_agent import run_book_rag_agent
from app.agents.judge_agent import judge_agent_answer
from app.agents.synthesis_agent import synthesize_final_answer
from app.agents.youtube_agent import run_youtube_agent
from app.config import get_settings
from app.graph.state import JudgeResult, MusicResearchState


def _choose_route(question: str) -> tuple[str, str]:
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


def supervisor_router_node(state: MusicResearchState) -> MusicResearchState:
    route_decision, route_reason = _choose_route(state["user_question"])
    return {
        **state,
        "route_decision": route_decision,
        "route_reason": route_reason,
        "retry_count": state.get("retry_count", 0),
        "errors": state.get("errors", []),
    }


def book_agent_node(state: MusicResearchState) -> MusicResearchState:
    retry_instruction = None
    if state.get("retry_target") == "books" and state.get("book_judgement"):
        retry_instruction = state["book_judgement"].retry_instruction
    answer = run_book_rag_agent(state["user_question"], retry_instruction=retry_instruction)
    return {**state, "book_answer": answer, "retry_target": None}


def youtube_agent_node(state: MusicResearchState) -> MusicResearchState:
    retry_instruction = None
    if state.get("retry_target") == "youtube" and state.get("youtube_judgement"):
        retry_instruction = state["youtube_judgement"].retry_instruction
    answer = run_youtube_agent(state["user_question"], retry_instruction=retry_instruction)
    return {**state, "youtube_answer": answer, "retry_target": None}


def judge_node(state: MusicResearchState) -> MusicResearchState:
    updated_state = dict(state)
    if state.get("book_answer"):
        updated_state["book_judgement"] = judge_agent_answer(
            state["user_question"], state["book_answer"]
        )
    if state.get("youtube_answer"):
        updated_state["youtube_judgement"] = judge_agent_answer(
            state["user_question"],
            state["youtube_answer"],
        )
    return updated_state


def _pick_retry_target(state: MusicResearchState) -> str | None:
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


def retry_router_node(state: MusicResearchState) -> MusicResearchState:
    target = _pick_retry_target(state)
    if target is None:
        return {**state, "retry_target": None}
    return {
        **state,
        "retry_count": state.get("retry_count", 0) + 1,
        "retry_target": target,
    }


def synthesis_node(state: MusicResearchState) -> MusicResearchState:
    final_answer = synthesize_final_answer(
        question=state["user_question"],
        book_answer=state.get("book_answer"),
        youtube_answer=state.get("youtube_answer"),
        book_judgement=state.get("book_judgement"),
        youtube_judgement=state.get("youtube_judgement"),
    )
    return {**state, "final_answer": final_answer}


def route_after_supervisor(state: MusicResearchState) -> str:
    if state["route_decision"] == "books":
        return "book_agent_node"
    if state["route_decision"] == "youtube":
        return "youtube_agent_node"
    return "book_agent_node"


def route_after_book(state: MusicResearchState) -> str:
    if state.get("route_decision") == "both" and not state.get("youtube_answer"):
        return "youtube_agent_node"
    return "judge_node"


def route_after_youtube(state: MusicResearchState) -> str:
    return "judge_node"


def route_after_judge(state: MusicResearchState) -> str:
    settings = get_settings()
    retry_target = _pick_retry_target(state)
    if retry_target and state.get("retry_count", 0) < settings.max_agent_retries:
        return "retry_router_node"
    return "synthesis_node"


def route_after_retry(state: MusicResearchState) -> str:
    if state.get("retry_target") == "books":
        return "book_agent_node"
    return "youtube_agent_node"
