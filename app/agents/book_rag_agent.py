from __future__ import annotations

from textwrap import shorten

from app.config import get_settings
from app.graph.prompts import BOOK_AGENT_PROMPT
from app.graph.state import AgentAnswer
from app.llm_client import MissingLLMConfigurationError, generate_text
from app.rag.retriever import retrieve_book_chunks


def _build_fallback_answer(
    question: str, chunks: list[dict], retry_instruction: str | None
) -> AgentAnswer:
    if not chunks:
        return AgentAnswer(
            agent_name="book_rag_agent",
            answer=(
                "I could not answer from the local library because no relevant document chunks "
                "were retrieved. Run ingestion first or add more source material."
            ),
            citations=[],
            confidence="low",
            limitations=["No retrieved local-document evidence was available."],
            metadata={"mode": "fallback", "prompt": BOOK_AGENT_PROMPT},
        )

    evidence_lines = []
    for item in chunks[:3]:
        evidence_lines.append(
            f"- {item['book_title']}: {shorten(item['chunk_text'], width=220, placeholder='...')}"
        )

    retry_line = f" Judge guidance: {retry_instruction}" if retry_instruction else ""
    answer = (
        f"Based on the retrieved local-library evidence, the question "
        f"'{question}' points to these grounded ideas:\n"
        + "\n".join(evidence_lines)
        + "\nThis answer is a deterministic summary because no LLM provider is configured."
        + retry_line
    )
    citations = [item["citation"] for item in chunks[:3]]
    return AgentAnswer(
        agent_name="book_rag_agent",
        answer=answer,
        citations=citations,
        confidence="medium",
        limitations=["LLM synthesis is disabled because no LLM provider is enabled."],
        metadata={"mode": "fallback", "retrieved_chunks": len(chunks), "prompt": BOOK_AGENT_PROMPT},
    )


def _build_llm_answer(
    question: str, chunks: list[dict], retry_instruction: str | None
) -> AgentAnswer:
    try:
        context = "\n\n".join(
            f"{item['citation']['label']}\n{item['chunk_text']}" for item in chunks[:4]
        )
        answer_text = generate_text(
            system_prompt=BOOK_AGENT_PROMPT,
            user_prompt=(
                f"Question: {question}\n\nRetrieved context:\n{context}\n\n"
                f"Retry guidance: {retry_instruction or 'none'}\n\n"
                "Write a concise answer with explicit grounding and no unsupported claims."
            ),
        )
    except MissingLLMConfigurationError:
        return _build_fallback_answer(question, chunks, retry_instruction)

    citations = [item["citation"] for item in chunks[:4]]
    return AgentAnswer(
        agent_name="book_rag_agent",
        answer=answer_text,
        citations=citations,
        confidence="high" if citations else "low",
        limitations=[],
        metadata={"mode": "llm", "retrieved_chunks": len(chunks), "prompt": BOOK_AGENT_PROMPT},
    )


def run_book_rag_agent(question: str, retry_instruction: str | None = None) -> AgentAnswer:
    chunks = retrieve_book_chunks(question)
    settings = get_settings()
    if not settings.llm_enabled:
        return _build_fallback_answer(question, chunks, retry_instruction)
    return _build_llm_answer(question, chunks, retry_instruction)
