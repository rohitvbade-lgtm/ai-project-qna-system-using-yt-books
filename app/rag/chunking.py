from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PageChunk:
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slice_with_overlap(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    if not words:
        return chunks

    while start < len(words):
        current_words: list[str] = []
        current_len = 0
        index = start
        while index < len(words):
            word = words[index]
            projected = current_len + len(word) + (1 if current_words else 0)
            if current_words and projected > chunk_size:
                break
            current_words.append(word)
            current_len = projected
            index += 1

        chunks.append(" ".join(current_words))
        if index >= len(words):
            break

        overlap_chars = 0
        overlap_words = 0
        reverse_index = len(current_words) - 1
        while reverse_index >= 0 and overlap_chars < overlap:
            overlap_chars += len(current_words[reverse_index]) + 1
            overlap_words += 1
            reverse_index -= 1
        start = max(start + 1, index - overlap_words)

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    words = normalized.split()
    return _slice_with_overlap(words, chunk_size=chunk_size, overlap=overlap)


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int, overlap: int) -> list[PageChunk]:
    word_entries: list[tuple[str, int]] = []
    for page_number, page_text in pages:
        normalized = clean_text(page_text)
        if not normalized:
            continue
        for word in normalized.split():
            word_entries.append((word, page_number))

    if not word_entries:
        return []

    chunks: list[PageChunk] = []
    start = 0
    chunk_index = 0
    while start < len(word_entries):
        current_entries: list[tuple[str, int]] = []
        current_len = 0
        index = start
        while index < len(word_entries):
            word, page = word_entries[index]
            projected = current_len + len(word) + (1 if current_entries else 0)
            if current_entries and projected > chunk_size:
                break
            current_entries.append((word, page))
            current_len = projected
            index += 1

        text = " ".join(word for word, _ in current_entries)
        pages_used = [page for _, page in current_entries]
        chunks.append(
            PageChunk(
                chunk_index=chunk_index,
                text=text,
                page_start=min(pages_used) if pages_used else None,
                page_end=max(pages_used) if pages_used else None,
            )
        )

        if index >= len(word_entries):
            break

        overlap_chars = 0
        overlap_words = 0
        reverse_index = len(current_entries) - 1
        while reverse_index >= 0 and overlap_chars < overlap:
            overlap_chars += len(current_entries[reverse_index][0]) + 1
            overlap_words += 1
            reverse_index -= 1
        start = max(start + 1, index - overlap_words)
        chunk_index += 1

    return chunks
