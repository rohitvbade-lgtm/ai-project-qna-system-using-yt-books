from app.config import get_settings
from app.graph.nodes import supervisor_router_node


def test_book_prompt_routes_to_books():
    state = supervisor_router_node(
        {"user_question": "According to the books I uploaded, explain photosynthesis."}
    )
    assert state["route_decision"] == "books"


def test_youtube_prompt_routes_to_youtube():
    state = supervisor_router_node(
        {"user_question": "Find YouTube explanations of black holes."}
    )
    assert state["route_decision"] == "youtube"


def test_comparison_prompt_routes_to_both():
    state = supervisor_router_node(
        {"user_question": "Compare YouTube explanations with the books for photosynthesis."}
    )
    assert state["route_decision"] == "both"


def test_supervisor_uses_llm_routing_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "test-groq-key")
    get_settings.cache_clear()

    monkeypatch.setattr(
        "app.graph.nodes.generate_text",
        lambda **_: (
            '{"route_decision":"youtube","route_reason":"The user explicitly asked for '
            'a video explanation."}'
        ),
    )

    state = supervisor_router_node({"user_question": "Find a video explanation of black holes."})
    assert state["route_decision"] == "youtube"
    assert state["route_reason"] == "The user explicitly asked for a video explanation."
