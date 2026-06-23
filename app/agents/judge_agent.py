from __future__ import annotations

from collections import Counter

from app.graph.state import AgentAnswer, JudgeResult


def _tokenize(text_value: str) -> list[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in text_value.split() if token.strip()]


def judge_agent_answer(question: str, answer: AgentAnswer) -> JudgeResult:
    question_tokens = Counter(_tokenize(question))
    answer_tokens = Counter(_tokenize(answer.answer))
    answer_word_count = len(answer.answer.split())
    overlap = sum(min(count, answer_tokens[token]) for token, count in question_tokens.items())

    relevance = min(1.0, overlap / max(len(question_tokens), 1))
    source_support = 0.9 if answer.citations else 0.0
    if answer.citations and len(answer.answer.split()) > 80:
        source_support = max(source_support, 0.95)
    citation_quality = min(1.0, len(answer.citations) / 3)
    completeness = min(1.0, len(answer.answer.split()) / 160)

    hallucination_risk = "low"
    if not answer.citations or answer.confidence == "high" and len(answer.citations) < 2:
        hallucination_risk = "high"
    elif answer.confidence == "high" and "metadata only" in " ".join(answer.limitations).lower():
        hallucination_risk = "medium"

    failures: list[str] = []
    if not answer.citations:
        failures.append("Add grounded citations.")
    simple_question = len(question_tokens) <= 3
    if answer_word_count < 80 and not (
        simple_question
        and len(answer.citations) >= 2
        and relevance >= 0.5
        and answer_word_count >= 35
    ):
        failures.append("Expand the answer with clearer evidence and explanation.")
    if answer.confidence == "high" and len(answer.citations) < 2:
        failures.append("Lower confidence or strengthen source support.")
    if source_support < 0.5:
        failures.append("Support the answer with stronger retrieved evidence.")
    if relevance < 0.3:
        failures.append("Address the user question more directly.")

    passed = not failures
    summary = "Pass" if passed else "Fail: " + " ".join(failures)
    retry_instruction = (
        "Keep the answer tightly aligned to the question, cite retrieved "
        "evidence, and reduce unsupported confidence."
        if failures
        else "No retry needed."
    )
    return JudgeResult(
        relevance_to_question=round(relevance, 2),
        source_support=round(source_support, 2),
        citation_quality=round(citation_quality, 2),
        completeness=round(completeness, 2),
        hallucination_risk=hallucination_risk,
        passed=passed,
        retry_instruction=retry_instruction,
        summary=summary,
    )
