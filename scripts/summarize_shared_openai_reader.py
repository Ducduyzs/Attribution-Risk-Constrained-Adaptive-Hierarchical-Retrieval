"""Validate and summarize the shared GPT-4o controlled-reader experiment."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SYSTEMS = ("B3_flat_neural", "B4_static_hierarchy", "edahr_prior", "B5_raptor_faithful")
INPUT_USD_PER_MILLION = 2.50
OUTPUT_USD_PER_MILLION = 10.00
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-4o"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from edahr.baselines import BenchmarkRun, clustered_ci_vs_baseline, significance_vs_baseline

    run_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main" / "shared_gpt4o_reader"
    manifest_path = PROJECT_ROOT / "manifests" / "qasper_baseline_dev_questions.jsonl"
    manifest = load_jsonl(manifest_path)
    expected_ids = {str(row["question_id"]) for row in manifest}
    runs: dict[str, BenchmarkRun] = {}
    summary: dict = {
        "task": "shared GPT-4o reader over leaf-normalized frozen contexts",
        "num_questions": len(manifest),
        "pricing": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "source": PRICING_URL,
            "accessed_date": dt.date.today().isoformat(),
        },
        "systems": {},
        "comparisons_vs_B3": {},
        "manifest_sha256": sha256_file(manifest_path),
    }

    for system in SYSTEMS:
        path = run_dir / f"{system}_rows.jsonl"
        rows = load_jsonl(path)
        ids = [str(row["question_id"]) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise RuntimeError(f"{system}: coverage is not exactly the frozen manifest")
        errors = load_jsonl(run_dir / f"{system}_errors.jsonl")
        if errors:
            raise RuntimeError(f"{system}: {len(errors)} recorded errors")
        finish_reasons = sorted({str(row.get("finish_reason")) for row in rows})
        if finish_reasons != ["stop"]:
            raise RuntimeError(f"{system}: unexpected finish reasons {finish_reasons}")
        runs[system] = BenchmarkRun(system, rows, {})
        prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
        completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
        latencies = [float(row["latency_ms"]) for row in rows]
        estimated_cost = (
            prompt_tokens * INPUT_USD_PER_MILLION
            + completion_tokens * OUTPUT_USD_PER_MILLION
        ) / 1_000_000
        summary["systems"][system] = {
            "num_queries": len(rows),
            "failures": 0,
            "answer_f1": statistics.fmean(float(row["answer_f1"]) for row in rows),
            "official_qasper_evidence_f1": statistics.fmean(
                float(row["official_qasper_evidence_f1"]) for row in rows
            ),
            "mean_prompt_tokens": prompt_tokens / len(rows),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "mean_latency_ms": statistics.fmean(latencies),
            "response_models": sorted({str(row["response_model"]) for row in rows}),
            "estimated_api_cost_usd": estimated_cost,
            "rows_sha256": sha256_file(path),
        }

    baseline = runs["B3_flat_neural"]
    baseline_f1 = summary["systems"]["B3_flat_neural"]["answer_f1"]
    for system in SYSTEMS[1:]:
        low, high = clustered_ci_vs_baseline(
            runs[system], baseline, metric="answer_f1", seed=42
        )
        summary["comparisons_vs_B3"][system] = {
            "answer_f1_delta": summary["systems"][system]["answer_f1"] - baseline_f1,
            "paired_p": significance_vs_baseline(
                runs[system], baseline, metric="answer_f1", seed=42
            ),
            "paper_clustered_95ci": [low, high],
        }

    summary["total_estimated_api_cost_usd"] = sum(
        values["estimated_api_cost_usd"] for values in summary["systems"].values()
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    report = [
        "# Shared GPT-4o controlled reader",
        "",
        "All systems use the same prompt and GPT-4o reader over leaf-normalized candidate contexts.",
        "",
        "| System | N | Answer F1 | Evidence F1 | Mean input tokens | API cost (USD) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        values = summary["systems"][system]
        report.append(
            f"| {system} | {values['num_queries']} | {values['answer_f1']:.4f} | "
            f"{values['official_qasper_evidence_f1']:.4f} | "
            f"{values['mean_prompt_tokens']:.1f} | {values['estimated_api_cost_usd']:.4f} |"
        )
    report += ["", "## Paired answer-F1 comparisons versus B3", ""]
    for system, values in summary["comparisons_vs_B3"].items():
        low, high = values["paper_clustered_95ci"]
        report.append(
            f"- {system}: delta={values['answer_f1_delta']:.4f}, "
            f"p={values['paired_p']:.4f}, paper-clustered 95% CI [{low:.4f}, {high:.4f}]."
        )
    report += [
        "",
        f"Estimated total API token charge: ${summary['total_estimated_api_cost_usd']:.4f}.",
        "No superiority claim is warranted when the paired confidence interval contains zero.",
    ]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
