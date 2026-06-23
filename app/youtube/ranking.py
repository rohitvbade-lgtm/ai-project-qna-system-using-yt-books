from __future__ import annotations

from collections import Counter
from typing import Any


def _tokenize(text_value: str) -> list[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in text_value.split() if token.strip()]


def rank_videos_for_question(
    question: str,
    videos: list[dict[str, Any]],
    transcripts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    question_counts = Counter(_tokenize(question))
    ranked: list[dict[str, Any]] = []

    for video in videos:
        transcript = transcripts.get(video["video_id"], {})
        transcript_text = " ".join(
            segment.get("text", "") for segment in transcript.get("segments", [])
        )
        haystack = " ".join([video["title"], video["description"], transcript_text]).lower()
        score = sum(count for token, count in question_counts.items() if token in haystack)
        if transcript.get("status") == "available":
            score += 3
        ranked.append({**video, "score": score, "transcript": transcript})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)
