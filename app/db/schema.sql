CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS books (
    id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NULL,
    file_name TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS book_chunks (
    id UUID PRIMARY KEY,
    book_id UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chapter TEXT NULL,
    page_start INT NULL,
    page_end INT NULL,
    chunk_text TEXT NOT NULL,
    token_count INT NULL,
    embedding VECTOR(768) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_book_chunks_chunk_index ON book_chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_book_chunks_embedding_hnsw
    ON book_chunks
    USING hnsw (embedding vector_cosine_ops);
