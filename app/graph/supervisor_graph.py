from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.graph.nodes import (
    book_agent_node,
    judge_node,
    retry_router_node,
    route_after_book,
    route_after_judge,
    route_after_retry,
    route_after_supervisor,
    route_after_youtube,
    supervisor_router_node,
    synthesis_node,
    youtube_agent_node,
)
from app.graph.state import MusicResearchState

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - dependency failure
    END = "__end__"
    START = "__start__"
    StateGraph = None


@dataclass
class FallbackCompiledGraph:
    def invoke(self, initial_state: MusicResearchState) -> MusicResearchState:
        state = supervisor_router_node(initial_state)
        next_node = route_after_supervisor(state)

        while True:
            if next_node == "book_agent_node":
                state = book_agent_node(state)
                next_node = route_after_book(state)
            elif next_node == "youtube_agent_node":
                state = youtube_agent_node(state)
                next_node = route_after_youtube(state)
            elif next_node == "judge_node":
                state = judge_node(state)
                next_node = route_after_judge(state)
            elif next_node == "retry_router_node":
                state = retry_router_node(state)
                next_node = route_after_retry(state)
            elif next_node == "synthesis_node":
                state = synthesis_node(state)
                return state
            else:
                raise RuntimeError(f"Unknown node transition: {next_node}")


def build_supervisor_graph() -> Any:
    if StateGraph is None:
        return FallbackCompiledGraph()

    workflow = StateGraph(MusicResearchState)
    workflow.add_node("supervisor_router_node", supervisor_router_node)
    workflow.add_node("book_agent_node", book_agent_node)
    workflow.add_node("youtube_agent_node", youtube_agent_node)
    workflow.add_node("judge_node", judge_node)
    workflow.add_node("retry_router_node", retry_router_node)
    workflow.add_node("synthesis_node", synthesis_node)

    workflow.add_edge(START, "supervisor_router_node")
    workflow.add_conditional_edges(
        "supervisor_router_node",
        route_after_supervisor,
        {
            "book_agent_node": "book_agent_node",
            "youtube_agent_node": "youtube_agent_node",
        },
    )
    workflow.add_conditional_edges(
        "book_agent_node",
        route_after_book,
        {
            "youtube_agent_node": "youtube_agent_node",
            "judge_node": "judge_node",
        },
    )
    workflow.add_conditional_edges(
        "youtube_agent_node",
        route_after_youtube,
        {"judge_node": "judge_node"},
    )
    workflow.add_conditional_edges(
        "judge_node",
        route_after_judge,
        {
            "retry_router_node": "retry_router_node",
            "synthesis_node": "synthesis_node",
        },
    )
    workflow.add_conditional_edges(
        "retry_router_node",
        route_after_retry,
        {
            "book_agent_node": "book_agent_node",
            "youtube_agent_node": "youtube_agent_node",
        },
    )
    workflow.add_edge("synthesis_node", END)
    return workflow.compile()


def run_supervisor(question: str) -> MusicResearchState:
    get_settings().apply_runtime_environment()
    graph = build_supervisor_graph()
    return graph.invoke(
        {
            "user_question": question,
            "retry_count": 0,
            "errors": [],
        }
    )
