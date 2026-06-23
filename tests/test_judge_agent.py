from app.agents.judge_agent import judge_agent_answer
from app.graph.state import AgentAnswer


def test_no_citations_fails():
    result = judge_agent_answer(
        "Explain photosynthesis.",
        AgentAnswer(
            agent_name="book", answer="Short unsupported answer.", citations=[], confidence="high"
        ),
    )
    assert result.passed is False


def test_weak_short_answer_fails():
    result = judge_agent_answer(
        "Explain black holes.",
        AgentAnswer(
            agent_name="youtube",
            answer="Black holes are interesting.",
            citations=[{"label": "video 1"}],
            confidence="medium",
        ),
    )
    assert result.passed is False


def test_cited_relevant_answer_passes():
    answer = AgentAnswer(
        agent_name="book",
        answer=(
            "Photosynthesis is the process plants use to convert light energy "
            "into chemical energy. In broad terms, chlorophyll helps capture "
            "sunlight, water is split during the light-dependent reactions, "
            "and carbon dioxide is fixed into glucose during the Calvin cycle. "
            "That is why the process stores energy in sugar while releasing oxygen."
        ),
        citations=[
            {"label": "General Science Reader (p. 8)"},
            {"label": "General Science Reader (p. 9)"},
        ],
        confidence="medium",
    )
    result = judge_agent_answer("Explain photosynthesis.", answer)
    assert result.passed is True
