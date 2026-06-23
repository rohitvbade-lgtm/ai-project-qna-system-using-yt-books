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
