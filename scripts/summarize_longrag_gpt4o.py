"""Validate and summarize the official-reader LongRAG GPT-4o run."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

INPUT_USD_PER_MILLION = 2.50
OUTPUT_USD_PER_MILLION = 10.00
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-4o"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(proportion * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    from edahr.baselines import (
        BenchmarkRun,
        clustered_ci_vs_baseline,
        significance_vs_baseline,
    )

    output_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main"
    manifest_path = (
        PROJECT_ROOT / "manifests" / "qasper_baseline_dev_questions.jsonl")
    rows_path = output_dir / "B6_longrag_reader_gpt4o_rows.jsonl"
    traces_path = output_dir / "B6_longrag_reader_gpt4o_traces.jsonl"
    errors_path = output_dir / "errors_reader_gpt4o.jsonl"
    metadata_path = output_dir / "run_metadata_reader_gpt4o.json"
    baseline_path = output_dir / "B3_flat_neural_rows.jsonl"
    local_reader_path = output_dir / "B6_longrag_reader_rows.jsonl"

    questions = load_jsonl(manifest_path)
    rows = load_jsonl(rows_path)
    traces = load_jsonl(traces_path)
    errors = load_jsonl(errors_path)
    baseline_rows = load_jsonl(baseline_path)
    expected_ids = {str(row["question_id"]) for row in questions}
    row_ids = [str(row["question_id"]) for row in rows]
    trace_ids = [str(row["question_id"]) for row in traces]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != expected_ids:
        raise RuntimeError("GPT-4o prediction coverage is not exactly the manifest")
    if len(trace_ids) != len(set(trace_ids)) or set(trace_ids) != expected_ids:
        raise RuntimeError("GPT-4o trace coverage is not exactly the manifest")
    if errors:
        raise RuntimeError(f"GPT-4o run recorded {len(errors)} errors")
    finish_reasons = sorted({str(row.get("finish_reason")) for row in rows})
    if finish_reasons != ["stop"]:
        raise RuntimeError(f"Unexpected finish reasons: {finish_reasons}")

    run = BenchmarkRun("B6_longrag_reader_gpt4o", rows, {})
    baseline = BenchmarkRun("B3_flat_neural", baseline_rows, {})
    answer_scores = [float(row["answer_f1"]) for row in rows]
    evidence_scores = [
        float(row["official_qasper_evidence_f1"]) for row in rows]
    baseline_scores = [float(row["answer_f1"]) for row in baseline_rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    cached_tokens = sum(int(row["cached_prompt_tokens"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    uncached_tokens = prompt_tokens - cached_tokens
    estimated_cost = (
        uncached_tokens * INPUT_USD_PER_MILLION
        + cached_tokens * (INPUT_USD_PER_MILLION / 2)
        + completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    ci_low, ci_high = clustered_ci_vs_baseline(
        run, baseline, metric="answer_f1", seed=42)

    summary = {
        "system": "B6_longrag_reader_gpt4o",
        "classification": "LongRAG reported reader configuration",
        "requested_model": "gpt-4o",
        "response_models": sorted({str(row["response_model"]) for row in rows}),
        "num_queries": len(rows),
        "failures": 0,
        "finish_reasons": finish_reasons,
        "answer_f1": statistics.fmean(answer_scores),
        "official_qasper_evidence_f1": statistics.fmean(evidence_scores),
        "b3_answer_f1": statistics.fmean(baseline_scores),
        "answer_f1_delta_vs_b3": (
            statistics.fmean(answer_scores) - statistics.fmean(baseline_scores)),
        "paired_p_vs_b3_answer_f1": significance_vs_baseline(
            run, baseline, metric="answer_f1", seed=42),
        "paper_clustered_95ci_delta_vs_b3_answer_f1": [ci_low, ci_high],
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "total": sum(latencies),
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "estimated_api_cost_usd": estimated_cost,
        "pricing": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "cached_input_usd_per_million": INPUT_USD_PER_MILLION / 2,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "source": PRICING_URL,
            "accessed_date": dt.date.today().isoformat(),
        },
        "provenance": {
            "predictions_sha256": sha256_file(rows_path),
            "traces_sha256": sha256_file(traces_path),
            "question_manifest_sha256": sha256_file(manifest_path),
            "run_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "execution_log": "artifacts/baselines/main/execution_reader_gpt4o.log",
        },
    }

    if local_reader_path.is_file():
        local_rows = load_jsonl(local_reader_path)
        question_by_id = {
            str(question["question_id"]): question for question in questions}
        enriched_local = [
            {
                **row,
                "source": question_by_id[str(row["question_id"])].get("source"),
            }
            for row in local_rows
        ]
        local_run = BenchmarkRun("B6_longrag_reader_local", enriched_local, {})
        local_scores = [float(row["answer_f1"]) for row in enriched_local]
        local_ci_low, local_ci_high = clustered_ci_vs_baseline(
            run, local_run, metric="answer_f1", seed=42)
        summary["local_reader_answer_f1"] = statistics.fmean(local_scores)
        summary["answer_f1_delta_vs_local_reader"] = (
            summary["answer_f1"] - statistics.fmean(local_scores))
        summary["paired_p_vs_local_reader_answer_f1"] = (
            significance_vs_baseline(
                run, local_run, metric="answer_f1", seed=42))
        summary["paper_clustered_95ci_delta_vs_local_reader_answer_f1"] = [
            local_ci_low, local_ci_high]

    summary_path = output_dir / "B6_longrag_reader_gpt4o_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    report = [
        "# LongRAG GPT-4o reader summary",
        "",
        f"- Valid predictions/traces: {len(rows)}/75; failures: 0.",
        f"- API-resolved model: {', '.join(summary['response_models'])}.",
        f"- Answer token F1: {summary['answer_f1']:.4f}.",
        f"- Official QASPER evidence F1: "
        f"{summary['official_qasper_evidence_f1']:.4f}.",
        f"- Answer-F1 delta vs B3: {summary['answer_f1_delta_vs_b3']:.4f}; "
        f"paired p={summary['paired_p_vs_b3_answer_f1']:.4f}; "
        f"paper-clustered 95% CI [{ci_low:.4f}, {ci_high:.4f}].",
        f"- Tokens: {prompt_tokens:,} input + {completion_tokens:,} output "
        f"= {total_tokens:,} total.",
        f"- Latency: mean {summary['latency_ms']['mean'] / 1000:.2f}s, "
        f"median {summary['latency_ms']['median'] / 1000:.2f}s, "
        f"p95 {summary['latency_ms']['p95'] / 1000:.2f}s.",
        f"- Estimated token charge: ${estimated_cost:.4f} using the OpenAI "
        "price recorded in the JSON summary.",
    ]
    (output_dir / "reader_gpt4o_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
