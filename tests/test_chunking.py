from app.rag.chunking import chunk_pages, chunk_text


def test_chunk_text_chunks_are_non_empty():
    chunks = chunk_text("Photosynthesis converts light into stored chemical energy." * 10, 80, 20)
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_overlap_works():
    chunks = chunk_text(
        "one two three four five six seven eight nine ten eleven twelve",
        22,
        10,
    )
    assert len(chunks) >= 2
    assert any(word in chunks[1] for word in chunks[0].split()[-3:])


def test_chunk_size_approximately_respected():
    chunks = chunk_text("lorem ipsum dolor sit amet " * 30, 100, 15)
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_page_chunking_preserves_page_ranges():
    chunks = chunk_pages([(1, "alpha beta gamma " * 10), (2, "delta epsilon " * 10)], 80, 20)
    assert chunks
    assert chunks[0].page_start in {1, 2}
    assert chunks[-1].page_end in {1, 2}
