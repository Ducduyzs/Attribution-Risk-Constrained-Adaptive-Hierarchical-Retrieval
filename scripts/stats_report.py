"""Statistical comparison table: proposed system vs every baseline.

Per metric and baseline: mean paired difference, paper-cluster bootstrap 95%
CI, paired cluster sign-flip p-value, Holm-adjusted p over the whole family
(baselines x metrics), Cohen's d_z on per-paper differences, and a
non-inferiority test at ``--margin``. On the test split the margin and the
family must be fixed in the protocol before the run; on dev they are
exploratory.

Example (v8 dev):
  python scripts/stats_report.py --proposed prior \
    --run prior=artifacts/baselines/v8_dev_gpt4omini/artifacts_prior.jsonl \
    --run B3_flat_neural=artifacts/baselines/v8_fulldoc_dev/artifacts_B3_flat_neural.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.statistics import compare, holm  # noqa: E402

METRICS = {
    "citation_f1": "Citation F1",
    "official_qasper_evidence_f1": "Evidence F1",
    "answer_f1": "Answer F1",
}
V8_RUNS = {
    "prior": "artifacts/baselines/v8_dev_gpt4omini/artifacts_prior.jsonl",
    "B5_raptor_faithful": "artifacts/baselines/v8_dev_gpt4omini/artifacts_B5_raptor_faithful.jsonl",
    "B4_static_hierarchy": "artifacts/baselines/v8_dev_gpt4omini/artifacts_B4_static_hierarchy.jsonl",
    "B3_flat_neural": "artifacts/baselines/v8_fulldoc_dev/artifacts_B3_flat_neural.jsonl",
    "full_document": "artifacts/baselines/v8_fulldoc_dev/artifacts_full_document.jsonl",
    "B6_longrag_faithful": "artifacts/baselines/v8_dev_gpt4omini/artifacts_B6_longrag_faithful.jsonl",
}


def load(path: str) -> dict[str, dict]:
    text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    return {row["question_id"]: row for row in map(json.loads, text.splitlines()) if row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[],
                        help="name=path to a benchmark artifacts jsonl (default: v8 dev)")
    parser.add_argument("--proposed", default="prior")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--label", default="v8 dev, 75 questions / 30 papers (exploratory)")
    parser.add_argument("--output", default="analysis/stats_v8_dev")
    args = parser.parse_args()

    paths = dict(item.split("=", 1) for item in args.run) if args.run else V8_RUNS
    runs = {name: load(path) for name, path in paths.items()}
    proposed = runs[args.proposed]
    clusters = {qid: str(row["source"]) for qid, row in proposed.items()}
    results = []
    for baseline, rows in runs.items():
        if baseline == args.proposed:
            continue
        for metric in METRICS:
            outcome = compare(
                {q: r.get(metric) for q, r in proposed.items()},
                {q: r.get(metric) for q, r in rows.items()},
                clusters, margin=args.margin,
            )
            results.append({"baseline": baseline, "metric": metric, **outcome})
    for result, adjusted in zip(results, holm([r["p"] for r in results])):
        result["p_holm"] = adjusted

    lines = [
        f"# {args.proposed} vs baselines — {args.label}", "",
        f"Family of {len(results)} comparisons, Holm-adjusted. Paper = sampling unit. "
        f"Non-inferiority margin {args.margin} (absolute).", "",
        "| Baseline | Metric | Diff | 95% CI (paper bootstrap) | p | p (Holm) | d_z "
        f"| Non-inferior @{args.margin} |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for r in results:
        ni = r["non_inferiority"]
        lines.append(
            f"| {r['baseline']} | {METRICS[r['metric']]} | {r['mean_diff']:+.3f} | "
            f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.3f} | {r['p_holm']:.3f} | "
            f"{r['d_z']:+.2f} | {'yes' if ni['non_inferior'] else 'no'} (p={ni['p']:.3f}) |"
        )
    output = PROJECT_ROOT / args.output
    output.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
