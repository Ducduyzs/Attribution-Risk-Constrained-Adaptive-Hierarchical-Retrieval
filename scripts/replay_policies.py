"""Evaluate verifier/citation policies offline on stored benchmark rows.

Policies (see edahr.replay):
  legacy          verifier as run (lexical guard vetoes NLI entailment)
  fixed           guard only gates the lexical fallback (current code)
  fixed+cite_retrieved   H4: expansion is read-only, cite retrieved leaves only
  fixed+crc@a     H5: expanded-leaf threshold chosen by Conformal Risk Control
                  so that expected harmful drift per paper <= a; cross-fitted
                  over paper folds (calibrate on k-1 folds, evaluate on 1).

The replay is first checked to reproduce every stored selection under
``legacy``. Answer F1 uses the refreshed (official) references of the
manifest; official evidence F1 maps cited leaves to their QASPER paragraphs
by rebuilding the run's hierarchy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.config import Settings  # noqa: E402
from edahr.evaluation import (  # noqa: E402
    citation_f1, citation_precision, citation_recall, paired_cluster_bootstrap,
    qasper_answer_token_f1, qasper_evidence_f1,
)
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402
from edahr.replay import INF, ReplayPolicy, replay_row  # noqa: E402
from edahr.risk_control import check_monotone, crc_threshold, paper_folds  # noqa: E402

GRID = [round(0.25 + 0.05 * step, 2) for step in range(16)]  # 0.25 .. 1.00


def load_context(args):
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    fields = Settings.__dataclass_fields__
    settings = replace(Settings(), **{k: v for k, v in config.items() if k in fields})
    papers = read_jsonl(PROJECT_ROOT / args.papers)
    hierarchy = HierarchyBuilder(settings).build(documents_from_paper_records(papers))
    questions = {row["question_id"]: row for row in read_jsonl(PROJECT_ROOT / args.questions)}
    base = ReplayPolicy(
        nli_support_threshold=settings.nli_support_threshold,
        nli_contradiction_threshold=settings.nli_contradiction_threshold,
        lexical_support_min_coverage=settings.lexical_support_min_coverage,
        evidence_margin=settings.evidence_margin,
        max_evidence_per_claim=settings.max_evidence_per_claim,
        sibling_delta=settings.sibling_threshold_delta,
    )
    return hierarchy, questions, base


def score_row(row: dict, out: dict, hierarchy, question: dict) -> dict:
    gold = set(row.get("gold_child_ids") or ())
    cited = out["cited_leaves"]
    paragraphs: dict[str, str] = {}
    for leaf in cited:
        node = hierarchy.nodes.get(leaf)
        if node is None:
            raise KeyError(f"leaf {leaf} not in rebuilt hierarchy; settings differ from the run")
        paragraphs.update(node.metadata.get("paragraph_texts") or {})
    answer = " ".join(out["claims"]).strip() or "Unanswerable"
    evaluable = bool(row.get("citation_evaluable"))
    return {
        "paper": str(row.get("source")),
        "citation_precision": citation_precision(cited, gold) if evaluable else None,
        "citation_recall": citation_recall(cited, gold) if evaluable else None,
        "citation_f1": citation_f1(cited, gold) if evaluable else None,
        "evidence_f1": qasper_evidence_f1(
            list(paragraphs.values()), question["reference_evidence_sets"]
        ),
        "answer_f1": qasper_answer_token_f1(answer, question["reference_answers"]),
        "accepted_claims": len(out["claims"]),
        "cited_leaves": len(cited),
        "harmful_drift": len(out["harmful_drift"]),
        "rescued": len(out["rescued"]),
    }


def drift_loss(row: dict, base: ReplayPolicy):
    """Per-question loss: harmful drift leaves / leaves cited when every
    passing expanded leaf is admitted. Bounded in [0,1], non-increasing."""
    open_policy = replace(base, expanded_threshold=base.nli_support_threshold)
    denominator = max(1, len(replay_row(row, open_policy)["cited_leaves"]))

    def loss(lam: float) -> float:
        out = replay_row(row, replace(base, expanded_threshold=lam))
        return len(out["harmful_drift"]) / denominator
    return loss


def summarize(scores: list[dict]) -> dict:
    keys = ("citation_precision", "citation_recall", "citation_f1", "evidence_f1",
            "answer_f1", "accepted_claims", "cited_leaves", "harmful_drift", "rescued")
    summary = {}
    for key in keys:
        values = [s[key] for s in scores if s[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
    summary["harmful_drift_total"] = sum(s["harmful_drift"] for s in scores)
    summary["rescued_total"] = sum(s["rescued"] for s in scores)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", default="artifacts/baselines/main")
    parser.add_argument("--systems", nargs="+", default=[
        "B3_flat_neural", "B4_static_hierarchy", "edahr_prior",
        "B5_raptor_faithful", "B6_longrag_faithful_controlled",
    ])
    parser.add_argument("--config", default="artifacts/baselines/main/config.json")
    parser.add_argument("--papers", default="manifests/qasper_baseline_dev_papers.jsonl")
    parser.add_argument("--questions", default="manifests/qasper_baseline_dev_questions.jsonl")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.02, 0.05, 0.10])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="analysis/replay_policies.json")
    args = parser.parse_args()

    hierarchy, questions, base = load_context(args)
    fixed = replace(base, guard="fixed")
    report: dict = {"config": args.config, "questions_manifest": args.questions, "systems": {}}
    for system in args.systems:
        rows = [json.loads(line) for line in
                (PROJECT_ROOT / args.rows_dir / f"{system}_rows.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]
        legacy = replace(base, guard="legacy")
        mismatch = sum(
            replay_row(row, legacy)["cited_leaves"] != sorted(set(row["evidence_node_ids"]))
            for row in rows
        )
        if mismatch:
            raise RuntimeError(f"{system}: replay does not reproduce {mismatch} stored rows")

        def run(policy_for_row) -> list[dict]:
            return [
                score_row(row, replay_row(row, policy_for_row(row)), hierarchy,
                          questions[row["question_id"]])
                for row in rows
            ]

        policies = {
            "legacy": run(lambda row: legacy),
            "fixed": run(lambda row: fixed),
            "fixed+cite_retrieved": run(lambda row: replace(fixed, expanded_threshold=INF)),
        }
        losses = {row["question_id"]: drift_loss(row, fixed) for row in rows}
        monotone = all(check_monotone(loss, GRID) for loss in losses.values())
        crc_detail = {}
        for alpha in args.alphas:
            lam_by_paper: dict[str, float] = {}
            for fold in paper_folds([str(row["source"]) for row in rows], args.folds):
                calibration: dict[str, list] = {}
                for row in rows:
                    if str(row["source"]) not in fold:
                        calibration.setdefault(str(row["source"]), []).append(
                            losses[row["question_id"]])
                unit_losses = [
                    (lambda fs: (lambda lam: sum(f(lam) for f in fs) / len(fs)))(fs)
                    for fs in calibration.values()
                ]
                lam = crc_threshold(unit_losses, GRID, alpha)
                for paper in fold:
                    lam_by_paper[paper] = lam
            name = f"fixed+crc@{alpha}"
            policies[name] = run(
                lambda row: replace(fixed, expanded_threshold=lam_by_paper[str(row["source"])])
            )
            crc_detail[name] = sorted({str(v) for v in lam_by_paper.values()})

        summaries = {name: summarize(scores) for name, scores in policies.items()}
        comparisons = {}
        reference = policies["legacy"]
        for name, scores in policies.items():
            if name == "legacy":
                continue
            comparisons[name] = {}
            for metric in ("citation_f1", "evidence_f1", "answer_f1"):
                pairs = [(s[metric], r[metric], s["paper"]) for s, r in zip(scores, reference)
                         if s[metric] is not None and r[metric] is not None]
                diffs = [a - b for a, b, _ in pairs]
                low, high = paired_cluster_bootstrap(diffs, [p for _, _, p in pairs])
                comparisons[name][metric] = {
                    "mean_diff": sum(diffs) / len(diffs) if diffs else None,
                    "paper_cluster_ci95": [low, high],
                }
        report["systems"][system] = {
            "rows": len(rows),
            "replay_reproduces_stored": True,
            "drift_loss_monotone": monotone,
            "crc_thresholds_by_fold": crc_detail,
            "summaries": summaries,
            "vs_legacy": comparisons,
        }
        print(f"\n== {system} ({len(rows)} rows, loss monotone={monotone})")
        print(f"{'policy':24} {'citF1':>6} {'citP':>6} {'citR':>6} {'evidF1':>6} "
              f"{'ansF1':>6} {'claims':>6} {'drift':>5} {'rescue':>6}")
        for name, s in summaries.items():
            print(f"{name:24} {s['citation_f1']:.4f} {s['citation_precision']:.4f} "
                  f"{s['citation_recall']:.4f} {s['evidence_f1']:.4f} {s['answer_f1']:.4f} "
                  f"{s['accepted_claims']:6.2f} {s['harmful_drift_total']:5d} "
                  f"{s['rescued_total']:6d}")
    output = PROJECT_ROOT / args.output
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {args.output}")


if __name__ == "__main__":
    main()
