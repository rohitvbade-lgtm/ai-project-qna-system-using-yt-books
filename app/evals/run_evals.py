from __future__ import annotations

from statistics import mean
from typing import Any

from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.evals.dataset import load_eval_dataset
from app.evals.graders import (
    answer_completeness,
    citation_presence,
    groundedness_heuristic,
    optional_llm_judge,
    route_accuracy,
)
from app.graph.supervisor_graph import run_supervisor
from app.runtime_logging import get_logger

console = Console()
logger = get_logger(__name__)


def _maybe_log_langsmith(results: list[dict[str, Any]]) -> str:
    settings = get_settings()
    if not settings.langsmith_enabled:
        return (
            "LangSmith tracing disabled. Set LANGSMITH_API_KEY to enable "
            "remote traces and eval visibility."
        )

    try:
        from langsmith import Client
    except ImportError:
        return "LangSmith SDK not installed."

    try:
        client = Client(api_key=settings.langsmith_api_key)
        _ = client
        return (
            f"LangSmith tracing is enabled for project '{settings.langsmith_project}'. "
            "Runs will appear when LangChain/LangGraph tracing hooks are active."
        )
    except Exception as exc:  # pragma: no cover - network/auth failure path
        return f"LangSmith setup failed: {exc}"


def run_local_evals() -> dict[str, Any]:
    get_settings().apply_runtime_environment()
    dataset = load_eval_dataset()
    if not dataset:
        return {"error": "No evaluation dataset found.", "results": []}

    logger.info("eval: running %s evaluation questions", len(dataset))
    results: list[dict[str, Any]] = []
    for index, item in enumerate(dataset, start=1):
        logger.info("eval: question %s/%s", index, len(dataset))
        state = run_supervisor(item["question"])
        final_answer = state.get("final_answer", "") or ""
        llm_grade = optional_llm_judge(final_answer)
        results.append(
            {
                "question": item["question"],
                "expected_route": item["expected_route"],
                "actual_route": state.get("route_decision"),
                "route_accuracy": route_accuracy(
                    item["expected_route"], state.get("route_decision", "")
                ),
                "citation_presence": citation_presence(final_answer),
                "answer_length": len(final_answer.split()),
                "judge_pass_rate": float(
                    any(
                        judgement and judgement.passed
                        for judgement in (
                            state.get("book_judgement"),
                            state.get("youtube_judgement"),
                        )
                    )
                ),
                "retry_used": float(state.get("retry_count", 0) > 0),
                "groundedness": groundedness_heuristic(state),
                "completeness": answer_completeness(final_answer),
                "llm_judge_score": llm_grade["score"],
            }
        )

    summary = {
        "route_accuracy": round(mean(item["route_accuracy"] for item in results), 2),
        "citation_presence_rate": round(mean(item["citation_presence"] for item in results), 2),
        "average_answer_length": round(mean(item["answer_length"] for item in results), 2),
        "judge_pass_rate": round(mean(item["judge_pass_rate"] for item in results), 2),
        "retry_rate": round(mean(item["retry_used"] for item in results), 2),
        "langsmith": _maybe_log_langsmith(results),
    }
    logger.info("eval: evaluation run complete")
    return {"summary": summary, "results": results}


def render_eval_report(report: dict[str, Any]) -> None:
    if "error" in report:
        console.print(f"[red]{report['error']}[/red]")
        return

    table = Table(title="General Agent Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in report["summary"].items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    render_eval_report(run_local_evals())
