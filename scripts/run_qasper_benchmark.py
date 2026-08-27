"""Run the frozen four-system benchmark on paper/question manifests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import (  # noqa: E402
    clustered_ci_vs_baseline,
    run_benchmark,
    significance_vs_baseline,
)
from edahr.config import Settings  # noqa: E402
from edahr.pipeline import AdaptiveHierarchicalPipeline  # noqa: E402
from edahr.policy import AdaptiveMergePolicy, NeverMergePolicy, StaticMergePolicy  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402
from edahr.runtime import build_pipeline_from_documents  # noqa: E402


def select_test(papers: list[dict], questions: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    evaluable = [row for row in questions if row.get("gold_quotes")]
    selected: list[dict] = []
    seen: set[str] = set()
    for row in evaluable:
        paper_id = str(row["paper_id"])
        if paper_id in seen:
            continue
        seen.add(paper_id)
        selected.append(row)
        if len(selected) >= limit:
            break
    used = {str(row["paper_id"]) for row in selected}
    return [row for row in papers if str(row["paper_id"]) in used], selected


def pipeline_like(base, settings, policy=None, parent_policy=None, section_policy=None):
    return AdaptiveHierarchicalPipeline(
        hierarchy=base.hierarchy, retriever=base.retriever,
        reranker=base.reranker, generator=base.generator, verifier=base.verifier,
        settings=settings, policy=policy, parent_policy=parent_policy,
        section_policy=section_policy, rerank_enabled=True,
    )


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def value(summary: dict, key: str) -> float:
    raw = summary.get(key)
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--paper-manifest", default="qasper_test_papers.jsonl")
    parser.add_argument("--question-manifest", default="qasper_test_questions.jsonl")
    parser.add_argument("--dataset-name", default="qasper-v0.3")
    parser.add_argument("--split-name", default="test-unseen-paper")
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--provider", default="openai",
                        choices=("openai", "gemini", "antigravity"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument(
        "--systems", nargs="+", default=("B_flat", "B_static", "prior", "learned_v5"),
        choices=("B_flat", "B_static", "prior", "learned_v5"),
    )
    parser.add_argument("--parent-checkpoint",
                        default="checkpoints/policy_parent_v5_final.joblib")
    parser.add_argument("--section-checkpoint",
                        default="checkpoints/policy_section_v5_final.joblib")
    parser.add_argument("--artifact-dir", default="data/artifacts/v5_final")
    parser.add_argument("--report", default="analysis/v5_main_results.md")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    config_path = Path(args.config)
    artifact_dir = Path(args.artifact_dir)
    report_path = Path(args.report)
    for name, path in (("manifest", manifest_dir), ("config", config_path),
                       ("artifact", artifact_dir), ("report", report_path)):
        if not path.is_absolute():
            resolved = PROJECT_ROOT / path
            if name == "manifest": manifest_dir = resolved
            elif name == "config": config_path = resolved
            elif name == "artifact": artifact_dir = resolved
            else: report_path = resolved

    papers, records = select_test(
        read_jsonl(manifest_dir / args.paper_manifest),
        read_jsonl(manifest_dir / args.question_manifest),
        args.questions,
    )
    settings = replace(
        Settings.from_json(config_path), llm_provider=args.provider,
        llm_model=args.model, policy_version="v5-tree",
        parent_policy_checkpoint=str(PROJECT_ROOT / args.parent_checkpoint),
        section_policy_checkpoint=str(PROJECT_ROOT / args.section_checkpoint),
    )
    learned = build_pipeline_from_documents(
        documents_from_paper_records(papers), settings
    )
    prior_settings = replace(
        settings, parent_policy_checkpoint=None, section_policy_checkpoint=None,
        policy_version="prior",
    )
    prior_policy = AdaptiveMergePolicy(
        threshold=prior_settings.merge_threshold,
        margin=prior_settings.merge_margin,
        evidence_gain_weight=prior_settings.evidence_gain_weight,
        cost_penalty=prior_settings.cost_penalty,
    )
    all_systems = {
        "B_flat": pipeline_like(
            learned, replace(prior_settings, expansion_max_depth=0),
            policy=NeverMergePolicy(),
        ),
        "B_static": pipeline_like(
            learned, prior_settings, policy=StaticMergePolicy(),
        ),
        "prior": pipeline_like(
            learned, prior_settings, parent_policy=prior_policy,
            section_policy=prior_policy,
        ),
        "learned_v5": learned,
    }
    systems = {
        name: all_systems[name]
        for name in dict.fromkeys(args.systems)
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, pipeline in systems.items():
        print(f"[benchmark] starting {name}", flush=True)
        run = run_benchmark(name, pipeline, records, seed=settings.seed)
        for row in run.rows:
            row["system"] = name
        write_jsonl(run.rows, artifact_dir / f"artifacts_{name}.jsonl")
        runs[name] = run
        print(json.dumps({"system": name, **run.summary}, indent=2), flush=True)

    comparisons = {}
    primary_metric = "official_qasper_evidence_f1"
    if "B_flat" in runs:
        flat = runs["B_flat"]
        for name in runs:
            if name == "B_flat":
                continue
            comparisons[name] = {
                "official_qasper_evidence_f1_p_vs_flat": significance_vs_baseline(
                    runs[name], flat, primary_metric, settings.seed
                ),
                "official_qasper_evidence_f1_diff_cluster_ci": clustered_ci_vs_baseline(
                    runs[name], flat, primary_metric, settings.seed
                ),
            }
    payload = {
        "dataset": args.dataset_name, "split": args.split_name,
        "questions": len(records), "papers": len(papers),
        "provider": args.provider, "model": args.model,
        "summaries": {name: run.summary for name, run in runs.items()},
        "comparisons_vs_flat": comparisons,
    }
    (artifact_dir / "main_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        f"# QASPER results — {args.dataset_name} {args.split_name}", "",
        f"Questions/papers: {len(records)}/{len(papers)}. Generator: `{args.provider}:{args.model}`.",
        "Configured thresholds and estimator families are recorded in the run provenance.", "",
        "| system | official evidence F1 | leaf attribution F1 | answer F1 | evidence-span recall | context tokens | latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        s = run.summary
        lines.append(
            f"| {name} | {value(s, 'official_qasper_evidence_f1'):.4f} | "
            f"{value(s, 'citation_f1'):.4f} | {value(s, 'answer_f1'):.4f} | "
            f"{value(s, 'evidence_span_recall'):.4f} | "
            f"{value(s, 'context_tokens'):.1f} | {value(s, 'latency_ms'):.1f} |"
        )
    if comparisons:
        lines.extend(["", "## Paired comparison against B_flat", ""])
        for name, comparison in comparisons.items():
            low, high = comparison["official_qasper_evidence_f1_diff_cluster_ci"]
            lines.append(
                f"- {name}: official evidence-F1 paired p="
                f"{comparison['official_qasper_evidence_f1_p_vs_flat']:.4f}; "
                f"paper-clustered 95% CI for difference [{low:.4f}, {high:.4f}]."
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
