from __future__ import annotations

from typing import Any

from app.youtube.search import YouTubeSearchClient
from app.youtube.transcripts import get_transcript


def search_youtube_videos(
    query: str,
    max_results: int = 5,
    require_captions: bool = True,
) -> list[dict[str, Any]]:
    client = YouTubeSearchClient()
    return client.search_videos(query, max_results=max_results, require_captions=require_captions)


def get_video_transcript(video_id: str) -> dict[str, Any]:
    return get_transcript(video_id)


def get_video_metadata(video_id: str) -> dict[str, Any] | None:
    client = YouTubeSearchClient()
    return client.get_video_metadata(video_id)


def build_server() -> Any:
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("fastmcp is not installed. Install project dependencies first.") from exc

    server = FastMCP("knowledge-youtube-tools")
    server.tool()(search_youtube_videos)
    server.tool()(get_video_transcript)
    server.tool()(get_video_metadata)
    return server


def main() -> None:
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
