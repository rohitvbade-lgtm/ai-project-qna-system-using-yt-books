from app.config import get_settings
from app.graph.supervisor_graph import run_supervisor


def test_full_graph_runs_with_fallback_data(seeded_library):
    state = run_supervisor("Compare YouTube explanations with the books for photosynthesis.")
    assert state["route_decision"] == "both"
    assert state["final_answer"]


def test_no_infinite_retry_loop(seeded_library):
    settings = get_settings()
    state = run_supervisor("Explain photosynthesis from my book library.")
    assert state["retry_count"] <= settings.max_agent_retries
    assert state["final_answer"]
