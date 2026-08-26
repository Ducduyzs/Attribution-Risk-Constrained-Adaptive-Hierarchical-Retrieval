"""Reproducible Rescue/HarmfulDrift decomposition for benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
from pathlib import Path


def decompose(row: dict) -> dict:
    retrieved = set(row.get("retrieved_child_ids") or [])
    verified = set(row.get("evidence_node_ids") or [])
    gold = set(row.get("gold_child_ids") or [])
    evaluable = bool(row.get("citation_evaluable", bool(gold))) and bool(gold)
    rescued = (verified - retrieved) & gold
    harmful = (verified - retrieved) - gold
    kept_correct = (verified & retrieved) & gold
    kept_wrong = (verified & retrieved) - gold
    return {
        "system": row.get("system", ""),
        "question_id": row.get("question_id", ""),
        "query": row.get("query", ""),
        "source": row.get("source", ""),
        "citation_evaluable": evaluable,
        "retrieved_count": len(retrieved),
        "verified_count": len(verified),
        "gold_count": len(gold),
        "rescued_leaf_ids": sorted(rescued),
        "harmful_drift_leaf_ids": sorted(harmful),
        "kept_correct_leaf_ids": sorted(kept_correct),
        "kept_wrong_leaf_ids": sorted(kept_wrong),
        "rescue_rate": len(rescued) / max(1, len(gold)) if evaluable else None,
        "harmful_drift_rate": len(harmful) / max(1, len(verified)) if evaluable else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="benchmark artifact JSONL files")
    parser.add_argument("--out-jsonl", default="data/artifacts/v5_decomposition.jsonl")
    parser.add_argument("--out-csv", default="data/artifacts/v5_drift_cases.csv")
    parser.add_argument("--report", default="analysis/v5_decomposition.md")
    args = parser.parse_args()

    rows: list[dict] = []
    provenance: list[tuple[str, str]] = []
    for item in args.inputs:
        path = Path(item)
        provenance.append((str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
        system = path.stem.removeprefix("artifacts_")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    raw = json.loads(line)
                    raw["system"] = system
                    rows.append(decompose(raw))

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    drift = [
        row for row in rows
        if row["citation_evaluable"]
        and (row["rescued_leaf_ids"] or row["harmful_drift_leaf_ids"])
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "system", "question_id", "query", "source", "gold_count",
        "rescued_leaf_ids", "harmful_drift_leaf_ids", "rescue_rate",
        "harmful_drift_rate",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in drift:
            serial = dict(row)
            serial["rescued_leaf_ids"] = "|".join(row["rescued_leaf_ids"])
            serial["harmful_drift_leaf_ids"] = "|".join(row["harmful_drift_leaf_ids"])
            writer.writerow(serial)

    evaluable = [row for row in rows if row["citation_evaluable"]]
    identified = [row for row in rows if row["question_id"]]
    systems = sorted({str(row["system"]) for row in rows})
    lines = [
        "# V5 Attribution Decomposition",
        "",
        "> Diagnostic decomposition of existing artifacts. It is not the final v5 result table",
        "> unless every evaluated row has a stable question ID and was generated after the",
        "> source-scoped retrieval, selector guard, and v5 reward fixes.",
        "",
        f"Total rows: {len(rows)}",
        f"Citation-evaluable rows: {len(evaluable)} ({len(evaluable) / max(1, len(rows)):.1%})",
        f"Rows with stable question IDs: {len(identified)} ({len(identified) / max(1, len(rows)):.1%})",
        f"Drift/rescue cases: {len(drift)}",
        "",
        "## Input provenance",
        "",
        *[f"- `{path}` — SHA-256 `{digest}`" for path, digest in provenance],
        "",
        "## Aggregate decomposition",
        "",
        "| system | evaluable | mean rescue | mean harmful drift |",
        "|---|---:|---:|---:|",
    ]
    for system in systems:
        subset = [row for row in evaluable if row["system"] == system]
        rescue = sum(float(row["rescue_rate"] or 0.0) for row in subset) / max(1, len(subset))
        harmful = sum(float(row["harmful_drift_rate"] or 0.0) for row in subset) / max(1, len(subset))
        lines.append(f"| {system} | {len(subset)} | {rescue:.4f} | {harmful:.4f} |")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
