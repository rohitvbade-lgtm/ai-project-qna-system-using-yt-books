# System Detailed Walkthrough

This document describes how the repository works today, based on the current implementation in `app/`, not the original prompt.

## 1. What This Project Actually Is

The repository is currently implemented as a CLI-first, backend-only multi-agent knowledge assistant with:

- a LangGraph supervisor flow
- a local document ingestion and retrieval pipeline
- a YouTube research pipeline using API data when available and local fixtures otherwise
- deterministic judge logic
- synthesis of one or two evidence paths into a final answer
- MCP-compatible tool servers for the library and YouTube layers
- optional LangSmith tracing and local evaluation commands

Important reality check:

- The repo naming and several prompts now describe a `general-purpose knowledge assistant`, not a strictly music-only system.
- The routing logic, RAG, YouTube layer, and evals are generic enough to answer non-music questions.
- The current `.env.example` defaults are set up for Ollama-style local inference rather than OpenAI defaults.

## 2. High-Level Runtime Architecture

The main runtime paths are:

1. `app/main.py`
2. `app/graph/supervisor_graph.py`
3. `app/graph/nodes.py`
4. one or both of:
   - `app/agents/book_rag_agent.py`
   - `app/agents/youtube_agent.py`
5. `app/agents/judge_agent.py`
6. `app/agents/synthesis_agent.py`

At a high level:

1. The CLI receives a question.
2. Settings are loaded from environment variables through `pydantic-settings`.
3. The supervisor decides whether the question should use:
   - books
   - youtube
   - both
4. The relevant agent or agents gather evidence and generate answers.
5. Each agent answer is scored by the judge.
6. If an answer is weak and retries remain, the weakest agent is rerun.
7. The synthesis stage builds the final answer.

## 3. Configuration and Startup

Configuration lives in [app/config.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/config.py).

Key behaviors:

- Settings are read from `.env` and environment variables.
- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL` control text generation.
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, and `EMBEDDING_BASE_URL` control embeddings.
- `DATABASE_URL` controls the database backend.
- `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` control observability.
- `MAX_AGENT_RETRIES`, `RETRIEVAL_TOP_K`, `CHUNK_SIZE`, and `CHUNK_OVERLAP` tune runtime behavior.

Runtime flags derived from config:

- `llm_enabled` is true only when a supported provider and usable API key are present.
- `embeddings_enabled` is true only when a supported embedding provider and usable API key are present.
- `langsmith_enabled` is true only when tracing is enabled and an API key exists.

Current LLM provider behavior:

- `ollama` uses `http://localhost:11434/v1` by default.
- `groq` uses `https://api.groq.com/openai/v1` by default.
- `openai` uses `LLM_BASE_URL` only when you explicitly set one.

`app/main.py` calls `get_settings().apply_runtime_environment()` before running commands so LangSmith-related environment variables are exported for downstream tooling.

## 4. CLI Commands

The CLI entrypoint is [app/main.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/main.py).

Implemented commands:

- `ask`
  - runs the full supervisor graph
- `ask-books`
  - runs only the local library agent
- `ask-library`
  - alias for `ask-books`
- `ask-youtube`
  - runs only the YouTube agent
- `ingest-books`
  - ingests PDFs from `data/books/raw`
- `ingest-library`
  - alias for `ingest-books`
- `eval`
  - runs local evaluation logic across the YAML dataset

Output formatting uses `rich` panels and tables.

## 5. Database Layer

The ORM models live in [app/db/models.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/db/models.py).

Tables:

- `books`
  - one row per source document
- `book_chunks`
  - one row per chunk of extracted text

Important current behavior after the pgvector fix:

- On PostgreSQL, `book_chunks.embedding` is now modeled as a real `vector` column through `pgvector`.
- On SQLite, the same field falls back to JSON storage so local tests and no-Postgres flows can still work.

Database initialization lives in [app/db/session.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/db/session.py).

Current behavior:

1. The engine is created from `DATABASE_URL`.
2. The connection is validated with `SELECT 1`.
3. If the URL is PostgreSQL and the server is unreachable, the app now raises a clear runtime error instead of silently switching to SQLite.
4. If PostgreSQL is reachable, the app attempts to enable `CREATE EXTENSION IF NOT EXISTS vector`.
5. SQLAlchemy creates the tables if missing.
6. The app validates that `book_chunks.embedding` is already a `vector` column on PostgreSQL and fails fast if the schema is outdated.
7. The app ensures an HNSW pgvector index exists on the embedding column.

Why this matters:

- Before this fix, the project stored embeddings as JSON and ranked them in Python.
- After this fix, PostgreSQL can be used as an actual vector store when the `vector` extension exists on the database instance.

## 6. PostgreSQL and pgvector Requirements

The project assumes you already have a reachable PostgreSQL instance and point `DATABASE_URL` at it.

Current operational rule:

- If your current PostgreSQL server does not have the `vector` extension installed, the app now fails fast with a clear message telling you to install the extension on that PostgreSQL instance.

## 7. Book Ingestion Pipeline

The ingestion logic lives in [app/rag/ingest_books.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/rag/ingest_books.py).

End-to-end ingestion flow:

1. `ingest_books_directory()` scans `data/books/raw` for `*.pdf`.
2. `extract_pages_from_pdf()` opens each PDF with `PyMuPDF`.
3. Text is extracted page by page.
4. `clean_text()` normalizes whitespace and null characters.
5. `chunk_pages()` creates page-aware chunks while keeping page ranges.
6. Token counts are estimated with `tiktoken` when available.
7. The embedding provider is resolved.
8. If embeddings are available, each chunk is embedded.
9. Existing rows for the same file are deleted and replaced.
10. The book and chunks are inserted into the database.

Fallback behavior:

- If no embedding provider is enabled, the chunks are still stored.
- In that case, retrieval falls back to keyword scoring later.

## 8. Chunking Logic

Chunking lives in [app/rag/chunking.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/rag/chunking.py).

Implemented functions:

- `clean_text(text)`
  - removes null bytes
  - collapses repeated spaces/tabs
  - compresses large newline runs
- `chunk_text(text, chunk_size, overlap)`
  - character-budget chunking over words
- `chunk_pages(pages, chunk_size, overlap)`
  - page-aware chunking that preserves `page_start` and `page_end`

Chunk overlap is done by backing up by enough trailing words to approximately preserve the requested overlap width in characters.

## 9. Embedding Layer

Embeddings live in [app/rag/embeddings.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/rag/embeddings.py).

Implemented providers:

- `OpenAICompatibleEmbeddingProvider`
  - used for OpenAI-style APIs and Ollama's OpenAI-compatible endpoint
- `FakeEmbeddingProvider`
  - deterministic hash-based embedding used for tests and fallback scenarios

Selection behavior:

- if a real embedding provider is configured, the real provider is used
- if `allow_fake=True` or `EMBEDDING_PROVIDER=fake`, fake embeddings can be used
- otherwise the app returns `None` and retrieval falls back to keywords

## 10. Retrieval Flow

Retrieval lives in [app/rag/retriever.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/rag/retriever.py).

The project now supports three retrieval strategies:

### 10.1 PostgreSQL pgvector Retrieval

Used when:

- the active database dialect is PostgreSQL
- embeddings are enabled

Flow:

1. embed the user question
2. query `book_chunks` rows with non-null embeddings
3. order by `embedding.cosine_distance(query_embedding)`
4. limit to `top_k`
5. attach book metadata and normalized citations

This is the main fix made in this pass.

### 10.2 In-Memory Cosine Retrieval

Used when:

- embeddings exist
- but the backend is not PostgreSQL with pgvector

Flow:

1. load chunks into Python
2. embed the user question
3. compute cosine similarity in process
4. return the best `top_k`

### 10.3 Keyword Fallback Retrieval

Used when:

- embeddings are unavailable

Flow:

1. tokenize question and chunk text
2. score overlap plus a simple density bonus
3. rank results
4. return the best `top_k`

Returned fields include:

- `chunk_id`
- `book_title`
- `page_start`
- `page_end`
- `chunk_text`
- `citation`

## 11. Citation Handling

Citation formatting lives in [app/rag/citations.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/rag/citations.py).

Book citations are normalized into dicts with:

- `source_type`
- `title`
- `page_start`
- `page_end`
- `chunk_id`
- `url`
- `label`

Example label style:

- `Book Title (p. 8)`
- `Book Title (pp. 8-9)`

## 12. YouTube Research Layer

The YouTube stack lives under `app/youtube/`.

### 12.1 Search

[app/youtube/search.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/youtube/search.py)

Behavior:

- If `YOUTUBE_API_KEY` exists, it tries the YouTube Data API.
- If the API fails or no key exists, it falls back to `data/youtube_fixtures/search_results.json`.
- Search results include:
  - title
  - channel
  - video_id
  - url
  - description
  - published_at
  - captions flag when known

### 12.2 Transcripts

[app/youtube/transcripts.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/youtube/transcripts.py)

Behavior:

1. Check for a local fixture file at `data/youtube_fixtures/<video_id>.txt`.
2. If found, convert lines into timestamped segments.
3. Otherwise try `youtube_transcript_api` if installed.
4. If transcript retrieval is unavailable or fails, return a structured `status: unavailable` response instead of crashing.

### 12.3 Ranking

[app/youtube/ranking.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/youtube/ranking.py)

Ranking logic:

- token overlap between the question and:
  - video title
  - description
  - transcript text
- transcript availability gets a score bonus

## 13. Agent Behaviors

### 13.1 Book RAG Agent

[app/agents/book_rag_agent.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/agents/book_rag_agent.py)

Flow:

1. retrieve chunks from the local library
2. if LLM is disabled:
   - build a deterministic summary from top chunks
   - mark confidence as `medium` or `low`
   - explain that LLM synthesis is disabled
3. if LLM is enabled:
   - send retrieved evidence into the configured chat model
   - return citations, answer text, and confidence

### 13.2 YouTube Agent

[app/agents/youtube_agent.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/agents/youtube_agent.py)

Flow:

1. search videos
2. fetch transcripts or transcript status
3. rank results
4. if LLM is disabled:
   - summarize only retrieved metadata/transcript evidence
   - explicitly call out missing transcripts
5. if LLM is enabled:
   - send the evidence blocks to the LLM
   - return citations, confidence, and limitations

Important safety behavior:

- It does not invent transcript content when transcripts are unavailable.

### 13.3 Judge Agent

[app/agents/judge_agent.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/agents/judge_agent.py)

The judge is deterministic, not LLM-based.

It scores:

- relevance to question
- source support
- citation quality
- completeness
- hallucination risk

It fails answers when:

- citations are missing
- the answer is too short
- confidence is too high for the evidence quality
- source support is weak
- the answer does not address the question directly

### 13.4 Synthesis Agent

[app/agents/synthesis_agent.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/agents/synthesis_agent.py)

Behavior:

- If LLM is disabled, it assembles a structured deterministic final answer with sections for:
  - direct answer
  - local library explanation
  - YouTube explanation
  - agreement across sources
  - practical application
  - sources
  - confidence and limitations
- If LLM is enabled, it sends the gathered answers and judge summaries to the LLM for a final synthesized response.

## 14. Supervisor Graph

The graph is defined in [app/graph/supervisor_graph.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/graph/supervisor_graph.py) and [app/graph/nodes.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/graph/nodes.py).

### 14.1 State Shape

State is defined in [app/graph/state.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/graph/state.py).

Tracked fields include:

- `user_question`
- `route_decision`
- `route_reason`
- `youtube_answer`
- `book_answer`
- `youtube_judgement`
- `book_judgement`
- `retry_count`
- `retry_target`
- `final_answer`
- `errors`

### 14.2 Routing Rules

Current routing is LLM-based when an LLM provider is configured.

Primary behavior:

- the supervisor prompts the configured LLM for a JSON `route_decision` and `route_reason`
- the decision is normalized to one of `books`, `youtube`, or `both`
- the prompt tells the model to route by semantic intent instead of raw keyword matching
- if the routing call is unavailable or malformed, the supervisor falls back to deterministic rules

Fallback rules:

- mentions of `youtube`, `video`, `videos`, or `transcript` route toward `youtube`
- mentions of `uploaded`, `book`, `books`, `document`, `pdf`, or `library` route toward `books`
- mentions of `both`, `compare`, `contrast`, `versus`, or `vs` route toward `both`
- otherwise the default is `books`

### 14.3 Node Sequence

The effective graph is:

1. `supervisor_router_node`
2. route to:
   - `book_agent_node`
   - `youtube_agent_node`
   - or both sequentially
3. `judge_node`
4. optional `retry_router_node`
5. `synthesis_node`

### 14.4 Retry Logic

The retry system:

1. checks which judged answer failed
2. picks the weakest failing answer
3. increments `retry_count`
4. reruns only that agent
5. stops retrying when `MAX_AGENT_RETRIES` is reached

The current graph also has a fallback compiled implementation in pure Python if `langgraph` is not importable.

## 15. MCP Servers

There are two MCP-style server entrypoints.

### 15.1 Library MCP Server

[app/mcp_servers/library_server.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/mcp_servers/library_server.py)

Exposed tools:

- `search_book_chunks`
- `get_book_chunk_by_id`
- `list_available_books`

These wrap the existing retriever functions directly, so they are testable even outside MCP transport.

### 15.2 YouTube MCP Server

[app/mcp_servers/youtube_server.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/mcp_servers/youtube_server.py)

Exposed tools:

- `search_youtube_videos`
- `get_video_transcript`
- `get_video_metadata`

These wrap the YouTube search and transcript layers.

## 16. Evaluation System

The evaluation layer lives under `app/evals/`.

### 16.1 Dataset

[app/evals/dataset.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/evals/dataset.py) loads [app/evals/test_questions.yaml](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/evals/test_questions.yaml).

The current dataset has 12 questions:

- 4 expected `books`
- 4 expected `youtube`
- 4 expected `both`

### 16.2 Graders

[app/evals/graders.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/evals/graders.py)

Implemented graders:

- route accuracy
- citation presence
- answer completeness
- groundedness heuristic
- optional LLM judge

### 16.3 Eval Runner

[app/evals/run_evals.py](/D:/Projects/ai-project/ai-project-qna-system-with-yt-and-books/app/evals/run_evals.py)

Flow:

1. load the dataset
2. run the supervisor on each question
3. compute local metrics
4. render a `rich` summary table
5. optionally attempt LangSmith setup/logging

Summary metrics:

- `route_accuracy`
- `citation_presence_rate`
- `average_answer_length`
- `judge_pass_rate`
- `retry_rate`

## 17. LangSmith Behavior

The code is LangSmith-ready, but current behavior matters:

- `apply_runtime_environment()` exports LangSmith env vars.
- If tracing is enabled and a LangSmith API key is present, LangChain/LangGraph integrations may attempt remote tracing.
- In offline or restricted-network environments, those calls can add latency and produce connection errors.

This is operationally important for performance analysis.

## 18. Why a Simple Question Can Feel Slow

The current implementation can be slow for three separate reasons:

1. If `LLM_PROVIDER=ollama`, each answer path may call the local model.
2. The full `ask` path can make multiple model calls:
   - one in the book or YouTube agent
   - one more in synthesis
   - more if retries happen
3. If LangSmith tracing is enabled but the environment cannot reach LangSmith, failed network attempts can add delay.

After the pgvector fix:

- retrieval no longer has to load and rank all PostgreSQL chunks in Python when vector search is available
- but the system still pays for query embedding generation and LLM calls

So pgvector fixes correctness and scaling of retrieval, but it does not remove all latency by itself.

## 19. Current Strengths

The repository already does these things well:

- clean CLI entrypoints
- deterministic fallback behavior when LLMs are disabled
- structured answer payloads
- source-aware answer generation
- graceful YouTube fallback to fixtures
- deterministic judge and retry loop
- directly callable MCP tool functions
- local eval loop

## 20. Current Gaps and Limitations

Important current limitations in the codebase:

- the project is no longer narrowly music-specific in its implemented prompts and examples
- supervisor routing falls back to heuristics when no routing LLM is configured or the routing JSON is invalid
- retry guidance is generic, not source-specific
- YouTube transcript retrieval depends on fixtures or `youtube_transcript_api`
- synthesis still performs a second LLM call when LLM mode is enabled
- LangSmith can add latency in network-restricted environments
- PostgreSQL must actually have the `vector` extension installed to use the pgvector path

## 21. Practical Command Map

Useful runtime commands:

- `uv run python -m app.main ask "your question"`
- `uv run python -m app.main ask-books "your question"`
- `uv run python -m app.main ask-youtube "your question"`
- `uv run python -m app.main ingest-books`
- `uv run python -m app.main eval`
- `uv run python -m app.mcp_servers.library_server`
- `uv run python -m app.mcp_servers.youtube_server`

For pgvector-backed PostgreSQL:

1. start a PostgreSQL instance that includes the `vector` extension
2. point `DATABASE_URL` at that instance
3. make sure `book_chunks.embedding` is already using the `vector` type
4. run retrieval or `ask`

## 22. What Changed in This Pass

This analysis/fix pass changed the retrieval/storage path in an important way:

- added `pgvector` as a declared dependency
- changed PostgreSQL embeddings from JSON storage to `vector`
- removed automatic runtime migration and kept a fail-fast schema validation instead
- changed PostgreSQL retrieval to use SQL-side `cosine_distance`
- removed the silent PostgreSQL-to-SQLite fallback
- added a clear runtime error when PostgreSQL is present but the `vector` extension is missing

That means the project can now use PostgreSQL as an actual vector store, assuming the database instance supports pgvector.
