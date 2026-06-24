# General Knowledge Multi-Agent System

CLI-first knowledge assistant that combines a local PDF library, optional YouTube research, and configurable LLM synthesis. It is designed for general-purpose question answering, not a single domain, so you can ingest books or documents on any topic and ask grounded questions against that material.

## What It Does

A user asks a question in the CLI. A supervisor agent makes an LLM-based routing decision and sends the request to:

- the local library RAG agent
- the YouTube research agent
- or both when comparison is useful

Each sub-answer is judged for grounding and completeness, weak answers can be retried, and the final response includes citations, confidence, and limitations.

## Architecture

```text
User Question
    |
    v
Supervisor Agent
    |
    v
Route: Library / YouTube / Both
    |
    +------------------------+
    |                        |
    v                        v
Local Library Agent    YouTube Research Agent
    |                        |
    +-----------+------------+
                |
                v
        Judge / Evaluator
                |
        Retry if weak
                |
                v
        Final Synthesis
                |
                v
      Cited Knowledge Answer
```

## Default Local Stack

- Python 3.12+
- LangGraph for supervisor routing and retries
- PostgreSQL or SQLite fallback for local storage
- JSON-stored embeddings for model-agnostic local retrieval
- PyMuPDF for PDF parsing
- Typer + Rich for CLI UX
- FastMCP/MCP for tool exposure
- LangSmith for optional tracing/evaluation
- Ollama or Groq via OpenAI-compatible endpoints for LLM tasks

## Setup

1. Create `.env` from `.env.example`.
2. Create a PostgreSQL database on your local server and point `DATABASE_URL` at it.
3. The app creates its tables automatically on first use, so no Docker setup is required.
4. Install or start Ollama, then pull the models you want to use. Example:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

5. Sync dependencies:

```bash
uv sync
```

6. Verify the CLI:

```bash
uv run python -m app.main --help
```

## Environment

The example file is configured for a local Ollama setup:

- `LLM_PROVIDER=ollama`
- `LLM_BASE_URL=http://localhost:11434/v1`
- `EMBEDDING_PROVIDER=ollama`
- `EMBEDDING_BASE_URL=http://localhost:11434/v1`

If you want to use OpenAI instead, switch the providers to `openai` and provide the appropriate API keys.

If you want to use Groq for faster text generation, set:

- `LLM_PROVIDER=groq`
- `LLM_MODEL=<your groq model>`
- `LLM_API_KEY=<your groq api key>`

`LLM_BASE_URL` is optional for Groq. The app defaults it to `https://api.groq.com/openai/v1`.

Other important values:

- `DATABASE_URL`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `YOUTUBE_API_KEY`
- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `MAX_AGENT_RETRIES`
- `RETRIEVAL_TOP_K`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Example local database URL:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/AIProject
```

## Ingesting Documents

Place PDF books or documents in `data/books/raw/`, then run:

```bash
uv run python -m app.main ingest-library
```

Backward-compatible alias:

```bash
uv run python -m app.main ingest-books
```

If no embedding provider is enabled, the system still stores chunks and falls back to keyword retrieval.

## Asking Questions

Supervisor mode:

```bash
uv run python -m app.main ask "What caused the French Revolution?"
```

When an LLM provider is configured, the supervisor uses the model to choose `books`, `youtube`, or `both`.
If no LLM is configured or the routing response is malformed, it falls back to deterministic routing rules.

Local library only:

```bash
uv run python -m app.main ask-library "Explain photosynthesis from my uploaded documents."
```

Backward-compatible alias:

```bash
uv run python -m app.main ask-books "Explain photosynthesis from my uploaded documents."
```

YouTube only:

```bash
uv run python -m app.main ask-youtube "Find YouTube explanations of black holes."
```

Direct module entrypoints also work:

```bash
uv run python -m app.rag.ingest_books
uv run python -m app.rag.retriever "Summarize plate tectonics from my library."
```

## MCP Servers

Local library server:

```bash
uv run python -m app.mcp_servers.library_server
```

YouTube server:

```bash
uv run python -m app.mcp_servers.youtube_server
```

## Evaluation

```bash
uv run python -m app.main eval
```

Local metrics include:

- `route_accuracy`
- `citation_presence_rate`
- `average_answer_length`
- `judge_pass_rate`
- `retry_rate`

## Testing

```bash
uv run pytest
uv run ruff check .
```

Tests use SQLite, fake embeddings when needed, fixture-backed YouTube data, and no paid API keys.

## Current Limitations

- The YouTube layer avoids scraping and depends on API-backed metadata or local fixtures.
- Transcript retrieval still depends on either fixtures or an installed transcript dependency.
- The supervisor falls back to heuristic routing when no LLM is configured for routing.
- Retrieval uses in-process cosine similarity over stored embeddings, which keeps Ollama model choices flexible but is not optimized for very large libraries.
- LangSmith tracing only activates when credentials are configured.
