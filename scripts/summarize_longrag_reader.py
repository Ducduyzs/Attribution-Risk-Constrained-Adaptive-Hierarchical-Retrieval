"""Validate and summarize the resumed LongRAG local-reader artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from edahr.baselines import (
        BenchmarkRun,
        clustered_ci_vs_baseline,
        significance_vs_baseline,
    )

    output_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main"
    manifest_path = (
        PROJECT_ROOT / "manifests" / "qasper_baseline_dev_questions.jsonl")
    reader_path = output_dir / "B6_longrag_reader_rows.jsonl"
    baseline_path = output_dir / "B3_flat_neural_rows.jsonl"
    error_path = output_dir / "errors_reader_resume.jsonl"
    metadata_path = output_dir / "run_metadata_reader_resume.json"

    questions = load_jsonl(manifest_path)
    reader_rows = load_jsonl(reader_path)
    baseline_rows = load_jsonl(baseline_path)
    errors = load_jsonl(error_path)
    question_by_id = {str(row["question_id"]): row for row in questions}
    ids = [str(row.get("question_id")) for row in reader_rows]
    expected_ids = {str(row["question_id"]) for row in questions}
    if len(ids) != len(set(ids)):
        raise RuntimeError("Reader artifact contains duplicate question IDs")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))
        extra = sorted(set(ids) - expected_ids)
        raise RuntimeError(f"Reader coverage mismatch: missing={missing}, extra={extra}")
    if errors:
        raise RuntimeError(f"Reader resume recorded {len(errors)} errors")

    enriched = []
    for row in reader_rows:
        question = question_by_id[str(row["question_id"])]
        enriched.append({
            **row,
            "source": question.get("source") or question.get("paper_id"),
            "query": question.get("query"),
        })

    reader_run = BenchmarkRun("B6_longrag_reader_local", enriched, {})
    baseline_run = BenchmarkRun("B3_flat_neural", baseline_rows, {})
    answer_scores = [float(row["answer_f1"]) for row in enriched]
    evidence_scores = [float(row["evidence_f1"]) for row in enriched]
    baseline_answer = [float(row["answer_f1"]) for row in baseline_rows]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ci_low, ci_high = clustered_ci_vs_baseline(
        reader_run, baseline_run, metric="answer_f1", seed=42)
    summary = {
        "system": "B6_longrag_reader_local",
        "reader_model": "Qwen/Qwen2.5-7B-Instruct",
        "reader_classification": "local substitution; not an official LongRAG reader",
        "num_queries": len(enriched),
        "num_unique_questions": len(set(ids)),
        "failures": len(errors),
        "answer_f1": sum(answer_scores) / len(answer_scores),
        "official_qasper_evidence_f1": (
            sum(evidence_scores) / len(evidence_scores)),
        "b3_answer_f1": sum(baseline_answer) / len(baseline_answer),
        "answer_f1_delta_vs_b3": (
            sum(answer_scores) / len(answer_scores)
            - sum(baseline_answer) / len(baseline_answer)),
        "paired_p_vs_b3_answer_f1": significance_vs_baseline(
            reader_run, baseline_run, metric="answer_f1", seed=42),
        "paper_clustered_95ci_delta_vs_b3_answer_f1": [ci_low, ci_high],
        "gpu_peak_gb": metadata.get("gpu_peak_gb"),
        "latency": None,
        "context_tokens": None,
        "limitations": [
            "Recovery rows did not record per-query latency.",
            "Recovery rows did not record per-query context token counts.",
            "Qwen2.5-7B-Instruct is an offline local substitution for the "
            "paper's Gemini-1.5-Pro/GPT-4o reader.",
        ],
        "provenance": {
            "predictions_sha256": sha256_file(reader_path),
            "question_manifest_sha256": sha256_file(manifest_path),
            "resume_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "execution_log": "artifacts/baselines/main/execution_reader_resume.log",
        },
    }
    summary_path = output_dir / "B6_longrag_reader_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    report = [
        "# LongRAG local-reader recovery summary",
        "",
        f"- Valid predictions: {summary['num_queries']}/75; failures: 0.",
        f"- Answer token F1: {summary['answer_f1']:.4f}.",
        f"- Official QASPER paragraph evidence F1: "
        f"{summary['official_qasper_evidence_f1']:.4f}.",
        f"- Answer-F1 delta vs B3: {summary['answer_f1_delta_vs_b3']:.4f}; "
        f"paired p={summary['paired_p_vs_b3_answer_f1']:.4f}; "
        f"paper-clustered 95% CI [{ci_low:.4f}, {ci_high:.4f}].",
        f"- Peak allocated GPU memory: {summary['gpu_peak_gb']:.2f} GB.",
        "",
        "This is a controlled local-reader substitution, not the official "
        "Gemini-1.5-Pro/GPT-4o LongRAG reader. Per-query latency and context-token "
        "counts were not recorded by the interrupted/recovery run and remain null.",
    ]
    (output_dir / "reader_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
