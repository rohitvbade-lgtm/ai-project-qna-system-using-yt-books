from __future__ import annotations

from app.config import get_settings
from app.graph.prompts import SYNTHESIS_PROMPT
from app.graph.state import AgentAnswer, JudgeResult
from app.llm_client import MissingLLMConfigurationError, generate_text


def _accepted_block(
    heading: str,
    answer: AgentAnswer | None,
    judgement: JudgeResult | None,
) -> str:
    if not answer:
        return f"{heading}\nNo evidence gathered."
    status = "accepted" if judgement and judgement.passed else "included with caution"
    return f"{heading}\nStatus: {status}\n{answer.answer}"


def _fallback_synthesis(
    question: str,
    book_answer: AgentAnswer | None,
    youtube_answer: AgentAnswer | None,
    book_judgement: JudgeResult | None,
    youtube_judgement: JudgeResult | None,
) -> str:
    direct_answer_parts = []
    if book_answer and (not book_judgement or book_judgement.passed):
        direct_answer_parts.append(book_answer.answer)
    if youtube_answer and (not youtube_judgement or youtube_judgement.passed):
        direct_answer_parts.append(youtube_answer.answer)
    if not direct_answer_parts:
        direct_answer_parts = [
            answer.answer for answer in (book_answer, youtube_answer) if answer is not None
        ]

    sources: list[str] = []
    if book_answer:
        sources.extend(citation["label"] for citation in book_answer.citations)
    if youtube_answer:
        sources.extend(citation["label"] for citation in youtube_answer.citations)

    limitations = []
    for answer in (book_answer, youtube_answer):
        if answer:
            limitations.extend(answer.limitations)

    confidence = "medium"
    if all(
        judgement and judgement.passed
        for judgement in (book_judgement, youtube_judgement)
        if judgement
    ):
        confidence = "high"
    elif not any(
        judgement and judgement.passed for judgement in (book_judgement, youtube_judgement)
    ):
        confidence = "low"

    sections = [
        "Direct answer\n" + ("\n\n".join(direct_answer_parts) if direct_answer_parts else question),
        _accepted_block("Local library explanation", book_answer, book_judgement),
        _accepted_block("YouTube explanation", youtube_answer, youtube_judgement),
        "Agreement across sources\n"
        + (
            "Both sources broadly agree where they overlap."
            if book_answer and youtube_answer
            else "Only one source path produced usable evidence."
        ),
        "Practical application\n"
        "Apply the answer to a concrete example, worked problem, or follow-up question "
        "to verify the claim against the cited sources.",
        "Sources\n"
        + (
            "\n".join(f"- {source}" for source in sources) if sources else "- No sources available."
        ),
        f"Confidence and limitations\nConfidence: {confidence}\n"
        + (
            "\n".join(f"- {item}" for item in limitations)
            if limitations
            else "- No major limitations reported."
        ),
    ]
    return "\n\n".join(sections)


def synthesize_final_answer(
    question: str,
    book_answer: AgentAnswer | None,
    youtube_answer: AgentAnswer | None,
    book_judgement: JudgeResult | None,
    youtube_judgement: JudgeResult | None,
) -> str:
    settings = get_settings()
    if not settings.llm_enabled:
        return _fallback_synthesis(
            question,
            book_answer,
            youtube_answer,
            book_judgement,
            youtube_judgement,
        )

    try:
        return generate_text(
            system_prompt=SYNTHESIS_PROMPT,
            user_prompt=(
                f"Question: {question}\n\n"
                f"Local library answer: {book_answer.answer if book_answer else 'none'}\n"
                "Local library judgement: "
                f"{book_judgement.summary if book_judgement else 'none'}\n\n"
                f"YouTube answer: {youtube_answer.answer if youtube_answer else 'none'}\n"
                f"YouTube judgement: {youtube_judgement.summary if youtube_judgement else 'none'}"
            ),
        )
    except MissingLLMConfigurationError:
        return _fallback_synthesis(
            question,
            book_answer,
            youtube_answer,
            book_judgement,
            youtube_judgement,
        )
