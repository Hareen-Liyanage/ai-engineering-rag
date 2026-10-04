"""Run a labeled comparison of hybrid retrieval and plain vector RAG."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from src.eval.scorer import exact_match, judge_answer
from src.generation.generator import generate_answer
from src.guardrails.guardrails import run_guardrails
from src.retrieval import assembler, vector_search
from src.retrieval.router import _get_llm, retrieve_and_answer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = ROOT / "eval_questions.jsonl"
DEFAULT_JSON_REPORT = ROOT / "reports" / "benchmark.json"
DEFAULT_MARKDOWN_REPORT = ROOT / "reports" / "benchmark.md"
QUESTION_TYPES = {"single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope"}


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict[str, Any]]:
    questions = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"id", "type", "question", "expected_answer"}.difference(item)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if item["type"] not in QUESTION_TYPES:
                raise ValueError(f"{path}:{line_number}: unknown type {item['type']!r}")
            if item["type"] == "out_of_scope" and item["expected_answer"] != "REFUSE":
                raise ValueError(f"{path}:{line_number}: out-of-scope reference must be REFUSE")
            questions.append(item)
    if not questions:
        raise ValueError(f"No evaluation questions found in {path}")
    return questions


def _vector_only_answer(question: str, llm: Any, embedder: Any = None) -> tuple[str, float]:
    started = time.perf_counter()
    pre_guard = run_guardrails(question, answer=None)
    if pre_guard.blocked:
        return pre_guard.answer, (time.perf_counter() - started) * 1000

    vector_hits = vector_search.search(question, embedder=embedder)
    context = assembler.assemble_context(vector_hits, [])
    retrieved_ids = [int(hit["id"]) for hit in vector_hits if hit.get("id") is not None]
    generation = generate_answer(question, context, retrieved_ids, llm=llm)
    guarded = run_guardrails(
        question,
        answer=generation.answer,
        context=context,
        retrieved_chunk_ids=retrieved_ids,
    )
    return guarded.answer, (time.perf_counter() - started) * 1000


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["system"], []).append(record)

    summary = {}
    for system, system_records in grouped.items():
        by_type: dict[str, list[dict[str, Any]]] = {}
        by_hops: dict[str, list[dict[str, Any]]] = {}
        for record in system_records:
            by_type.setdefault(record["type"], []).append(record)
            if record["hop_count"] is not None:
                by_hops.setdefault(str(record["hop_count"]), []).append(record)

        def accuracy(items: list[dict[str, Any]], key: str) -> float | None:
            scores = [item[key] for item in items if item[key] is not None]
            return round(sum(scores) / len(scores), 4) if scores else None

        def group_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "count": len(items),
                "exact_match_accuracy": accuracy(items, "exact_match"),
                "judge_accuracy": accuracy(items, "judge_correct"),
                "p95_latency_ms": round(_percentile95([item["latency_ms"] for item in items]) or 0, 2),
            }

        summary[system] = {
            "question_count": len(system_records),
            "exact_match_accuracy": accuracy(system_records, "exact_match"),
            "judge_accuracy": accuracy(system_records, "judge_correct"),
            "p95_latency_ms": round(_percentile95([item["latency_ms"] for item in system_records]) or 0, 2),
            "by_type": {name: group_metrics(items) for name, items in sorted(by_type.items())},
            "by_hop_count": {name: group_metrics(items) for name, items in sorted(by_hops.items())},
            "api_cost_per_query_usd": None,
        }
    return summary


def run_benchmark(
    questions: list[dict[str, Any]],
    *,
    llm: Any = None,
    embedder: Any = None,
    judge: bool = True,
) -> dict[str, Any]:
    model = llm or _get_llm()
    records = []
    for item in questions:
        for system in ("vector", "hybrid"):
            if system == "vector":
                answer, latency_ms = _vector_only_answer(item["question"], model, embedder)
            else:
                result = retrieve_and_answer(item["question"], llm=model, embedder=embedder)
                answer, latency_ms = result["answer"], float(result["latency_ms"])

            exact = exact_match(item["expected_answer"], answer)
            judged = None
            judge_score = None
            judge_reason = None
            if judge and not exact:
                judgment = judge_answer(item["question"], item["expected_answer"], answer, model)
                judged, judge_score, judge_reason = (
                    judgment["correct"], judgment["score"], judgment["reason"]
                )
            records.append(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "hop_count": item.get("hop_count"),
                    "system": system,
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "answer": answer,
                    "exact_match": exact,
                    "judge_correct": True if exact else judged,
                    "judge_score": 1.0 if exact else judge_score,
                    "judge_reason": judge_reason,
                    "latency_ms": round(latency_ms, 2),
                    "api_cost_usd": None,
                }
            )

    return {
        "question_count": len(questions),
        "systems": ["vector", "hybrid"],
        "judge_enabled": judge,
        "cost_note": "API and ingestion costs are not instrumented; null means unavailable, not zero.",
        "summary": _summarize(records),
        "results": records,
    }


def write_reports(
    report: dict[str, Any],
    json_path: Path = DEFAULT_JSON_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN_REPORT,
    ingestion_cost_usd: float | None = None,
) -> None:
    report["ingestion_cost_usd"] = ingestion_cost_usd
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Benchmark Results",
        "",
        f"Labeled questions: {report['question_count']} | Systems: vector-only vs hybrid",
        "",
        "| System | Exact match | LLM judge | p95 latency | API cost/query | Ingestion cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system, metrics in report["summary"].items():
        exact = metrics["exact_match_accuracy"]
        judged = metrics["judge_accuracy"]
        lines.append(
            f"| {system} | {exact if exact is not None else 'n/a'} | "
            f"{judged if judged is not None else 'n/a'} | {metrics['p95_latency_ms']:.2f} ms | "
            f"n/a | {ingestion_cost_usd if ingestion_cost_usd is not None else 'n/a'} |"
        )
    lines.extend([
        "", "## Accuracy by Type", "",
        "| System | Type | Count | Exact match | LLM judge | p95 latency |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for system, metrics in report["summary"].items():
        for kind, group in metrics["by_type"].items():
            lines.append(
                f"| {system} | {kind} | {group['count']} | {group['exact_match_accuracy']} | "
                f"{group['judge_accuracy']} | {group['p95_latency_ms']:.2f} ms |"
            )
    lines.extend([
        "", "## Accuracy by Hop Count", "",
        "| System | Hops | Count | Exact match | LLM judge | p95 latency |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for system, metrics in report["summary"].items():
        for hops, group in metrics["by_hop_count"].items():
            lines.append(
                f"| {system} | {hops} | {group['count']} | {group['exact_match_accuracy']} | "
                f"{group['judge_accuracy']} | {group['p95_latency_ms']:.2f} ms |"
            )
    lines.extend(["", report["cost_note"], ""])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--ingestion-cost-usd", type=float)
    parser.add_argument("--no-judge", action="store_true", help="Skip judging non-exact answers")
    args = parser.parse_args()
    report = run_benchmark(load_questions(args.questions), judge=not args.no_judge)
    write_reports(report, args.json_report, args.markdown_report, args.ingestion_cost_usd)
    print(f"Wrote {args.json_report} and {args.markdown_report}")


if __name__ == "__main__":
    main()