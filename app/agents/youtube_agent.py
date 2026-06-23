from __future__ import annotations

from textwrap import shorten

from app.config import get_settings
from app.graph.prompts import YOUTUBE_AGENT_PROMPT
from app.graph.state import AgentAnswer
from app.llm_client import MissingLLMConfigurationError, generate_text
from app.rag.citations import normalize_citation
from app.youtube.ranking import rank_videos_for_question
from app.youtube.search import YouTubeSearchClient
from app.youtube.transcripts import get_transcript


def _video_citation(video: dict) -> dict:
    return normalize_citation(
        {
            "source_type": "youtube",
            "title": video["title"],
            "url": video["url"],
            "label": f"{video['title']} - {video['channel']}",
        }
    )


def _fallback_answer(
    question: str, ranked_items: list[dict], retry_instruction: str | None
) -> AgentAnswer:
    if not ranked_items:
        return AgentAnswer(
            agent_name="youtube_agent",
            answer=(
                "I could not assemble a YouTube-based answer because no "
                "fixture-backed or API-backed videos "
                "were available for this question."
            ),
            citations=[],
            confidence="low",
            limitations=["No YouTube evidence was available."],
            metadata={"mode": "fallback", "prompt": YOUTUBE_AGENT_PROMPT},
        )

    evidence_lines = []
    limitations: list[str] = []
    citations: list[dict] = []
    for item in ranked_items[:3]:
        transcript = item.get("transcript", {})
        transcript_status = transcript.get("status")
        excerpt = " ".join(
            segment.get("text", "") for segment in transcript.get("segments", [])[:2]
        ).strip()
        if transcript_status != "available":
            limitations.append(
                f"Transcript unavailable for {item['title']}; relying on "
                "metadata only for that video."
            )
            excerpt = item["description"]
        evidence_lines.append(
            f"- {item['title']}: {shorten(excerpt, width=220, placeholder='...')}"
        )
        citations.append(_video_citation(item))

    retry_line = f" Judge guidance: {retry_instruction}" if retry_instruction else ""
    answer = (
        f"From the available YouTube evidence for '{question}', these sources "
        "are the strongest matches:\n"
        + "\n".join(evidence_lines)
        + "\nThis answer avoids unsupported transcript claims and stays within "
        "retrieved metadata/transcript evidence." + retry_line
    )
    confidence = (
        "medium"
        if any(item.get("transcript", {}).get("status") == "available" for item in ranked_items)
        else "low"
    )
    if not limitations:
        limitations.append("Results depend on available fixtures or API-backed transcript access.")
    return AgentAnswer(
        agent_name="youtube_agent",
        answer=answer,
        citations=citations,
        confidence=confidence,
        limitations=limitations,
        metadata={
            "mode": "fallback",
            "ranked_results": len(ranked_items),
            "prompt": YOUTUBE_AGENT_PROMPT,
        },
    )


def _llm_answer(
    question: str, ranked_items: list[dict], retry_instruction: str | None
) -> AgentAnswer:
    try:
        evidence_blocks = []
        citations: list[dict] = []
        for item in ranked_items[:4]:
            transcript = item.get("transcript", {})
            transcript_text = " ".join(
                segment.get("text", "") for segment in transcript.get("segments", [])[:6]
            )
            evidence_blocks.append(
                f"Video: {item['title']}\n"
                f"Channel: {item['channel']}\n"
                f"URL: {item['url']}\n"
                f"Transcript status: {transcript.get('status')}\n"
                f"Evidence: {transcript_text or item['description']}"
            )
            citations.append(_video_citation(item))

        answer_text = generate_text(
            system_prompt=YOUTUBE_AGENT_PROMPT,
            user_prompt=(
                f"Question: {question}\n\n"
                f"Retry guidance: {retry_instruction or 'none'}\n\n"
                + "\n\n".join(evidence_blocks)
            ),
        )
    except MissingLLMConfigurationError:
        return _fallback_answer(question, ranked_items, retry_instruction)

    limitations = []
    if not any(item.get("transcript", {}).get("status") == "available" for item in ranked_items):
        limitations.append("No transcript was available; the answer is based on metadata only.")
    return AgentAnswer(
        agent_name="youtube_agent",
        answer=answer_text,
        citations=citations,
        confidence="high" if citations else "low",
        limitations=limitations,
        metadata={
            "mode": "llm",
            "ranked_results": len(ranked_items),
            "prompt": YOUTUBE_AGENT_PROMPT,
        },
    )


def run_youtube_agent(question: str, retry_instruction: str | None = None) -> AgentAnswer:
    client = YouTubeSearchClient()
    videos = client.search_videos(question)
    transcripts = {video["video_id"]: get_transcript(video["video_id"]) for video in videos}
    ranked_items = rank_videos_for_question(question, videos, transcripts)
    settings = get_settings()
    if not settings.llm_enabled:
        return _fallback_answer(question, ranked_items, retry_instruction)
    return _llm_answer(question, ranked_items, retry_instruction)
