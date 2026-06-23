from __future__ import annotations

from pathlib import Path
from typing import Any

from app.runtime_logging import get_logger

logger = get_logger(__name__)


def get_transcript(video_id: str) -> dict[str, Any]:
    fixture_file = Path("data/youtube_fixtures") / f"{video_id}.txt"
    if fixture_file.exists():
        logger.info("transcript: loading transcript fixture for %s", video_id)
        text_value = fixture_file.read_text(encoding="utf-8").strip()
        segments = []
        for index, line in enumerate(text_value.splitlines()):
            cleaned = line.strip()
            if cleaned:
                segments.append({"start": index * 15, "text": cleaned})
        return {
            "video_id": video_id,
            "status": "available",
            "segments": segments,
            "source": "fixture",
        }

    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError:
        logger.info("transcript: no transcript dependency or fixture for %s", video_id)
        return {
            "video_id": video_id,
            "status": "unavailable",
            "segments": [],
            "source": "none",
            "error": "Transcript retrieval dependency not installed and no local fixture exists.",
        }

    try:
        logger.info("transcript: fetching transcript via API for %s", video_id)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return {
            "video_id": video_id,
            "status": "available",
            "segments": transcript,
            "source": "youtube_transcript_api",
        }
    except Exception as exc:
        logger.warning("transcript: transcript unavailable for %s: %s", video_id, exc)
        return {
            "video_id": video_id,
            "status": "unavailable",
            "segments": [],
            "source": "youtube_transcript_api",
            "error": f"Transcript unavailable: {exc}",
        }
