from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from app.agents.book_rag_agent import run_book_rag_agent
from app.agents.youtube_agent import run_youtube_agent
from app.evals.run_evals import render_eval_report, run_local_evals
from app.graph.supervisor_graph import run_supervisor
from app.rag.ingest_books import ingest_books_directory

app = typer.Typer(help="General-Purpose Knowledge Assistant CLI")
console = Console()


def _prepare_runtime() -> None:
    from app.config import get_settings

    get_settings().apply_runtime_environment()


def _render_agent_answer(title: str, payload) -> None:
    console.print(Panel.fit(payload.answer, title=title))
    citations = payload.citations or []
    if citations:
        console.print("[bold]Sources[/bold]")
        for citation in citations:
            console.print(f"- {citation['label']}")
    console.print(f"[bold]Confidence[/bold]\n{payload.confidence}")
    if payload.limitations:
        console.print("[bold]Limitations[/bold]")
        for limitation in payload.limitations:
            console.print(f"- {limitation}")


@app.command("ask")
def ask(question: str) -> None:
    _prepare_runtime()
    state = run_supervisor(question)
    console.print(
        Panel.fit(state.get("final_answer", "No answer generated."), title="Supervisor Answer")
    )
    console.print(
        f"[bold]Route[/bold]\n{state.get('route_decision')} - {state.get('route_reason')}"
    )
    console.print(f"[bold]Retries[/bold]\n{state.get('retry_count', 0)}")


def _ask_library(question: str) -> None:
    _prepare_runtime()
    answer = run_book_rag_agent(question)
    _render_agent_answer("Local Library Agent", answer)


@app.command("ask-books")
def ask_books(question: str) -> None:
    _ask_library(question)


@app.command("ask-library")
def ask_library(question: str) -> None:
    _ask_library(question)


@app.command("ask-youtube")
def ask_youtube(question: str) -> None:
    _prepare_runtime()
    answer = run_youtube_agent(question)
    _render_agent_answer("YouTube Agent", answer)


def _ingest_library() -> None:
    _prepare_runtime()
    results = ingest_books_directory(Path("data/books/raw"))
    if not results:
        console.print("[yellow]No PDF files found in data/books/raw.[/yellow]")
        return
    for item in results:
        console.print(
            Panel.fit(
                str(item),
                title=item.get("title", item.get("file_name", "Library ingestion")),
            )
        )


@app.command("ingest-books")
def ingest_books() -> None:
    _ingest_library()


@app.command("ingest-library")
def ingest_library() -> None:
    _ingest_library()


@app.command("eval")
def eval_command() -> None:
    _prepare_runtime()
    render_eval_report(run_local_evals())


if __name__ == "__main__":
    app()
