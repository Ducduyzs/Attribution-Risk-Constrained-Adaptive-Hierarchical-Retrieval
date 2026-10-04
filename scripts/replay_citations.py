"""Offline: how many leaves may one claim cite? (no API, no GPU)

The run used ``max_evidence_per_claim=1`` and kept only the top leaf when the
top-2 support gap is below ``evidence_margin``. Even with gold-only context
(oracle) citation recall was 0.54, so this cap may bound attribution recall
for every system. Each stored row is replayed from its verification trace
(``edahr.replay``) under k = 1, 2, 3, all, with and without the ambiguity
rule; the replay must first reproduce every stored selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from edahr.evaluation import paired_cluster_bootstrap  # noqa: E402
from edahr.replay import replay_row  # noqa: E402
from replay_policies import load_context, score_row, summarize  # noqa: E402

RUNS = {
    "oracle_evidence": "artifacts/baselines/v8_fulldoc_dev",
    "B3_flat_neural": "artifacts/baselines/v8_fulldoc_dev",
    "full_document": "artifacts/baselines/v8_fulldoc_dev",
    "B4_static_hierarchy": "artifacts/baselines/v8_dev_gpt4omini",
    "prior": "artifacts/baselines/v8_dev_gpt4omini",
    "B5_raptor_faithful": "artifacts/baselines/v8_dev_gpt4omini",
    "B6_longrag_faithful": "artifacts/baselines/v8_dev_gpt4omini",
}
VARIANTS = {
    "k1 (as run)": dict(max_evidence_per_claim=1),
    "k2": dict(max_evidence_per_claim=2),
    "k3": dict(max_evidence_per_claim=3),
    "k2 no-ambiguity-cut": dict(max_evidence_per_claim=2, evidence_margin=0.0),
    "k3 no-ambiguity-cut": dict(max_evidence_per_claim=3, evidence_margin=0.0),
    "all no-ambiguity-cut": dict(max_evidence_per_claim=0, evidence_margin=0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="artifacts/baselines/main/config.json")
    parser.add_argument("--papers", default="manifests/qasper_baseline_dev_papers.jsonl")
    parser.add_argument("--questions", default="manifests/qasper_baseline_dev_questions.jsonl")
    parser.add_argument("--output", default="analysis/replay_citations.json")
    args = parser.parse_args()

    hierarchy, questions, base = load_context(args)
    base = replace(base, guard="fixed")  # v8 runs used the fixed verifier
    report: dict = {}
    for system, directory in RUNS.items():
        path = PROJECT_ROOT / directory / f"artifacts_{system}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        mismatch = sum(
            replay_row(row, base)["cited_leaves"] != sorted(set(row["evidence_node_ids"]))
            for row in rows
        )
        if mismatch:
            raise RuntimeError(f"{system}: replay does not reproduce {mismatch} rows")
        scores = {
            name: [score_row(row, replay_row(row, replace(base, **change)), hierarchy,
                             questions[row["question_id"]]) for row in rows]
            for name, change in VARIANTS.items()
        }
        reference = scores["k1 (as run)"]
        system_report = {}
        print(f"\n== {system} (replay reproduces {len(rows)}/{len(rows)} rows)")
        print(f"{'variant':22} {'citP':>6} {'citR':>6} {'citF1':>6} {'evidF1':>6} "
              f"{'ansF1':>6} {'leaves':>6}  citF1 diff vs k1 [paper CI]")
        for name, values in scores.items():
            summary = summarize(values)
            pairs = [(v["citation_f1"], r["citation_f1"], v["paper"])
                     for v, r in zip(values, reference)
                     if v["citation_f1"] is not None and r["citation_f1"] is not None]
            diffs = [a - b for a, b, _ in pairs]
            low, high = paired_cluster_bootstrap(diffs, [p for _, _, p in pairs])
            summary["citation_f1_diff_vs_k1"] = sum(diffs) / len(diffs)
            summary["citation_f1_diff_ci95"] = [low, high]
            system_report[name] = summary
            print(f"{name:22} {summary['citation_precision']:.3f} "
                  f"{summary['citation_recall']:.3f} {summary['citation_f1']:.3f} "
                  f"{summary['evidence_f1']:.3f} {summary['answer_f1']:.3f} "
                  f"{summary['cited_leaves']:6.2f}  {summary['citation_f1_diff_vs_k1']:+.3f} "
                  f"[{low:+.3f},{high:+.3f}]")
        report[system] = system_report
    (PROJECT_ROOT / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {args.output}")


if __name__ == "__main__":
    main()
