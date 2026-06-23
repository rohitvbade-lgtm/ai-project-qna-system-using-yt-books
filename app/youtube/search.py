from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from app.config import get_settings
from app.runtime_logging import get_logger

FIXTURE_PATH = Path("data/youtube_fixtures/search_results.json")
logger = get_logger(__name__)


class YouTubeSearchClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def search_videos(
        self,
        query: str,
        max_results: int = 5,
        require_captions: bool = True,
    ) -> list[dict[str, Any]]:
        if self.settings.youtube_api_key:
            try:
                logger.info("youtube-search: searching YouTube API")
                return self._search_via_api(
                    query, max_results=max_results, require_captions=require_captions
                )
            except Exception as exc:
                logger.warning(
                    "youtube-search: API search failed, falling back to fixtures: %s",
                    exc,
                )
        logger.info("youtube-search: searching local fixture data")
        return self._search_via_fixtures(
            query, max_results=max_results, require_captions=require_captions
        )

    def get_video_metadata(self, video_id: str) -> dict[str, Any] | None:
        all_items = self._load_fixture_items()
        for item in all_items:
            if item["video_id"] == video_id:
                return item
        return None

    def _search_via_api(
        self,
        query: str,
        max_results: int,
        require_captions: bool,
    ) -> list[dict[str, Any]]:
        params = {
            "part": "snippet",
            "type": "video",
            "maxResults": max_results,
            "q": query,
            "key": self.settings.youtube_api_key,
        }
        if require_captions:
            params["videoCaption"] = "closedCaption"

        url = "https://www.googleapis.com/youtube/v3/search?" + urlencode(params)
        with urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        items: list[dict[str, Any]] = []
        for raw_item in payload.get("items", []):
            snippet = raw_item.get("snippet", {})
            video_id = raw_item.get("id", {}).get("videoId")
            if not video_id:
                continue
            items.append(
                {
                    "title": snippet.get("title", "Untitled video"),
                    "channel": snippet.get("channelTitle", "Unknown channel"),
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "description": snippet.get("description", ""),
                    "published_at": snippet.get("publishedAt"),
                    "captions_available": True if require_captions else None,
                }
            )
        return items

    def _search_via_fixtures(
        self,
        query: str,
        max_results: int,
        require_captions: bool,
    ) -> list[dict[str, Any]]:
        items = self._load_fixture_items()
        tokens = {token.lower() for token in query.split()}

        def _score(item: dict[str, Any]) -> int:
            haystack = " ".join([item["title"], item["description"], item["channel"]]).lower()
            score = sum(1 for token in tokens if token in haystack)
            if require_captions and item.get("captions_available"):
                score += 2
            return score

        ranked = sorted(
            (
                item
                for item in items
                if not require_captions or item.get("captions_available", False)
            ),
            key=_score,
            reverse=True,
        )
        results = ranked[:max_results]
        logger.info("youtube-search: fixture search returned %s videos", len(results))
        return results

    def _load_fixture_items(self) -> list[dict[str, Any]]:
        if not FIXTURE_PATH.exists():
            return []
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
