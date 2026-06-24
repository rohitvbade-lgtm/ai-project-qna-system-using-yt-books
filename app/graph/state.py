from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

CitationDict = dict[str, Any]


@dataclass
class AgentAnswer:
    agent_name: str
    answer: str
    citations: list[CitationDict] = field(default_factory=list)
    confidence: str = "low"
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    relevance_to_question: float
    source_support: float
    citation_quality: float
    completeness: float
    hallucination_risk: Literal["low", "medium", "high"]
    passed: bool
    retry_instruction: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneralResearchState(TypedDict, total=False):
    user_question: str
    route_decision: Literal["youtube", "books", "both"]
    route_reason: str
    youtube_answer: AgentAnswer | None
    book_answer: AgentAnswer | None
    youtube_judgement: JudgeResult | None
    book_judgement: JudgeResult | None
    retry_count: int
    retry_target: str | None
    final_answer: str | None
    errors: list[str]
