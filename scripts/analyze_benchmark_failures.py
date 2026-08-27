"""Classify benchmark failures from per-query retrieval and citation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def classify_failure(row: dict) -> str:
    gold = set(row.get("gold_child_ids") or [])
    evidence = set(row.get("evidence_node_ids") or [])
    candidates = set(row.get("candidate_child_ids") or [])
    if not row.get("citation_evaluable", bool(gold)) or not gold:
        return "not_evaluable"
    if float(row.get("hit_rate@10") or 0.0) == 0.0:
        return "retrieval_miss_at_10"
    if not candidates & gold:
        return "candidate_selection_miss"
    if not evidence:
        generated = row.get("generated_claim_count")
        if generated == 0:
            return "generation_empty"
        if isinstance(generated, int) and generated > 0:
            return "verifier_rejected_all"
        return "post_context_no_evidence"
    if not evidence & gold:
        return "wrong_citation_only"
    if gold <= evidence:
        return "full_grounding_success"
    return "partial_grounding_success"


def analyze_row(row: dict, system: str) -> dict:
    result = {
        "system": system,
        "question_id": str(row.get("question_id") or ""),
        "source": str(row.get("source") or ""),
        "query": str(row.get("query") or ""),
        "failure_class": classify_failure(row),
        "hit_rate@10": row.get("hit_rate@10"),
        "citation_f1": row.get("citation_f1"),
        "answer_f1": row.get("answer_f1"),
        "gold_count": len(set(row.get("gold_child_ids") or [])),
        "candidate_count": len(set(row.get("candidate_child_ids") or [])),
        "evidence_count": len(set(row.get("evidence_node_ids") or [])),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="benchmark artifact JSONL files")
    parser.add_argument("--out-csv", default="data/artifacts/v5_final/failure_cases.csv")
    parser.add_argument("--out-json", default="data/artifacts/v5_final/failure_summary.json")
    parser.add_argument("--report", default="analysis/v5_failure_analysis_fresh.md")
    args = parser.parse_args()

    rows: list[dict] = []
    provenance: list[dict] = []
    has_claim_counts = True
    for item in args.inputs:
        path = Path(item)
        system = path.stem.removeprefix("artifacts_")
        raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        has_claim_counts = has_claim_counts and all(
            "generated_claim_count" in row and "verified_claim_count" in row
            for row in raw_rows
        )
        rows.extend(analyze_row(row, system) for row in raw_rows)
        provenance.append({
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(raw_rows),
        })

    systems = sorted({row["system"] for row in rows})
    counts = {
        system: dict(sorted(Counter(
            row["failure_class"] for row in rows if row["system"] == system
        ).items()))
        for system in systems
    }
    payload = {
        "total_rows": len(rows), "has_claim_counts": has_claim_counts,
        "provenance": provenance, "counts": counts,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    classes = sorted({name for system_counts in counts.values() for name in system_counts})
    schema_note = (
        "Artifacts include generated and verified claim counts, so empty generation and "
        "verifier rejection are separated."
        if has_claim_counts else
        "`post_context_no_evidence` means gold reached the context candidate set but no "
        "verified evidence survived. These artifacts predate claim counters, so empty "
        "generation cannot be separated from verifier rejection."
    )
    lines = [
        "# Fresh V5 Failure Analysis", "",
        "This is a diagnostic analysis of the frozen 40-paper test. It must not be used",
        "to tune thresholds or select a replacement model.", "",
        schema_note, "",
        "## Input provenance", "",
        *[f"- `{item['path']}`: {item['rows']} rows, SHA-256 `{item['sha256']}`" for item in provenance],
        "", "## Failure counts", "",
        "| system | " + " | ".join(classes) + " | total |",
        "|---|" + "---:|" * (len(classes) + 1),
    ]
    for system in systems:
        system_counts = counts[system]
        total = sum(system_counts.values())
        lines.append(
            f"| {system} | "
            + " | ".join(str(system_counts.get(name, 0)) for name in classes)
            + f" | {total} |"
        )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()