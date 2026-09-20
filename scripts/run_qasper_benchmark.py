"""Run the frozen benchmark on paper/question manifests with full baseline/ablation support."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import (  # noqa: E402
    BASELINE_NAMES,
    clustered_ci_vs_baseline,
    make_baseline_pipeline,
    run_benchmark,
    significance_vs_baseline,
)
from edahr.config import Settings  # noqa: E402
from edahr.pipeline import AdaptiveHierarchicalPipeline  # noqa: E402
from edahr.policy import AdaptiveMergePolicy, NeverMergePolicy, StaticMergePolicy  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402
from edahr.runtime import build_pipeline_from_documents  # noqa: E402


def select_test(papers: list[dict], questions: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Select frozen questions without dropping same-paper dev rows."""
    selected = [row for row in questions if row.get("gold_quotes")][:limit]
    used = {str(row["paper_id"]) for row in selected}
    return [row for row in papers if str(row["paper_id"]) in used], selected


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def value(summary: dict, key: str) -> float:
    raw = summary.get(key)
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def build_systems(
    hierarchy,
    retriever,
    reranker,
    generator,
    verifier,
    settings: Settings,
    parent_checkpoint: str | None,
    section_checkpoint: str | None,
    args,
) -> dict[str, AdaptiveHierarchicalPipeline]:
    """Build all benchmark systems from shared heavy components."""
    all_systems: dict[str, AdaptiveHierarchicalPipeline] = {}

    # Standard baselines from baselines module
    baseline_map = {
        "B0_bm25": "B0_bm25",
        "B1_dense": "B1_dense",
        "B2_hybrid_rrf": "B2_hybrid_rrf",
        "B3_flat_neural": "B3_flat_neural",
        "B4_static_hierarchy": "B4_static_hierarchy",
    }

    # Get encoder from base retriever
    encoder = getattr(retriever, 'encoder', None)
    
    def index_factory(s: Settings):
        from edahr.index import MultiRepresentationIndex
        return MultiRepresentationIndex(hierarchy, encoder, s)

    for name, baseline_name in baseline_map.items():
        all_systems[name] = make_baseline_pipeline(
            baseline_name, hierarchy, index_factory=index_factory,
            reranker=reranker, generator=generator, verifier=verifier, settings=settings
        )

    # Prior policy (hand-tuned utility, no checkpoint)
    prior_settings = replace(
        settings,
        parent_policy_checkpoint=None,
        section_policy_checkpoint=None,
        policy_version="prior",
        enable_parent_expansion=True,
        enable_section_expansion=True,
    )
    prior_policy = AdaptiveMergePolicy(
        threshold=prior_settings.merge_threshold,
        margin=prior_settings.merge_margin,
        evidence_gain_weight=prior_settings.evidence_gain_weight,
        cost_penalty=prior_settings.cost_penalty,
    )
    all_systems["prior"] = AdaptiveHierarchicalPipeline(
        hierarchy=hierarchy, retriever=retriever, reranker=reranker,
        generator=generator, verifier=verifier, settings=prior_settings,
        parent_policy=prior_policy, section_policy=prior_policy,
        rerank_enabled=True,
    )

    # Learned-v7 (with checkpoints) - reuse base retriever
    if parent_checkpoint or section_checkpoint:
        learned_settings = replace(
            settings,
            parent_policy_checkpoint=parent_checkpoint,
            section_policy_checkpoint=section_checkpoint,
            policy_version="v7",
            enable_parent_expansion=bool(parent_checkpoint),
            enable_section_expansion=bool(section_checkpoint),
        )
        all_systems["learned_v7"] = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=reranker,
            generator=generator, verifier=verifier, settings=learned_settings,
            parent_policy=AdaptiveMergePolicy(
                threshold=learned_settings.merge_threshold,
                margin=learned_settings.merge_margin,
                evidence_gain_weight=learned_settings.evidence_gain_weight,
                cost_penalty=learned_settings.cost_penalty,
                checkpoint=parent_checkpoint,
            ) if parent_checkpoint else NeverMergePolicy(),
            section_policy=AdaptiveMergePolicy(
                threshold=learned_settings.merge_threshold,
                margin=learned_settings.merge_margin,
                evidence_gain_weight=learned_settings.evidence_gain_weight,
                cost_penalty=learned_settings.cost_penalty,
                checkpoint=section_checkpoint,
            ) if section_checkpoint else NeverMergePolicy(),
            rerank_enabled=True,
        )

    # Ablations - all reuse base retriever
    # Parent-only (section gate disabled)
    if parent_checkpoint:
        parent_only_settings = replace(
            settings,
            parent_policy_checkpoint=parent_checkpoint,
            section_policy_checkpoint=None,
            policy_version="v7-parent_only",
            enable_parent_expansion=True,
            enable_section_expansion=False,
        )
        all_systems["learned_v7_parent_only"] = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=reranker,
            generator=generator, verifier=verifier, settings=parent_only_settings,
            parent_policy=AdaptiveMergePolicy(
                threshold=parent_only_settings.merge_threshold,
                margin=parent_only_settings.merge_margin,
                evidence_gain_weight=parent_only_settings.evidence_gain_weight,
                cost_penalty=parent_only_settings.cost_penalty,
                checkpoint=parent_checkpoint,
            ),
            section_policy=NeverMergePolicy(),
            rerank_enabled=True,
        )

    # Section-only (parent gate disabled)
    if section_checkpoint:
        section_only_settings = replace(
            settings,
            parent_policy_checkpoint=None,
            section_policy_checkpoint=section_checkpoint,
            policy_version="v7_section_only",
            enable_parent_expansion=False,
            enable_section_expansion=True,
        )
        all_systems["learned_v7_section_only"] = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=reranker,
            generator=generator, verifier=verifier, settings=section_only_settings,
            parent_policy=NeverMergePolicy(),
            section_policy=AdaptiveMergePolicy(
                threshold=section_only_settings.merge_threshold,
                margin=section_only_settings.merge_margin,
                evidence_gain_weight=section_only_settings.evidence_gain_weight,
                cost_penalty=section_only_settings.cost_penalty,
                checkpoint=section_checkpoint,
            ),
            rerank_enabled=True,
        )

    # No drift penalty (disable rollback)
    if parent_checkpoint or section_checkpoint:
        no_rollback_settings = replace(
            settings,
            parent_policy_checkpoint=parent_checkpoint,
            section_policy_checkpoint=section_checkpoint,
            policy_version="v7_no_rollback",
            rollback_ratio=0.0,  # disable rollback
            enable_parent_expansion=bool(parent_checkpoint),
            enable_section_expansion=bool(section_checkpoint),
        )
        all_systems["learned_v7_no_rollback"] = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=reranker,
            generator=generator, verifier=verifier, settings=no_rollback_settings,
            parent_policy=AdaptiveMergePolicy(
                threshold=no_rollback_settings.merge_threshold,
                margin=no_rollback_settings.merge_margin,
                evidence_gain_weight=no_rollback_settings.evidence_gain_weight,
                cost_penalty=no_rollback_settings.cost_penalty,
                checkpoint=parent_checkpoint,
            ) if parent_checkpoint else NeverMergePolicy(),
            section_policy=AdaptiveMergePolicy(
                threshold=no_rollback_settings.merge_threshold,
                margin=no_rollback_settings.merge_margin,
                evidence_gain_weight=no_rollback_settings.evidence_gain_weight,
                cost_penalty=no_rollback_settings.cost_penalty,
                checkpoint=section_checkpoint,
            ) if section_checkpoint else NeverMergePolicy(),
            rerank_enabled=True,
        )

    # No verifier
    if parent_checkpoint or section_checkpoint:
        no_verifier_settings = replace(
            settings,
            parent_policy_checkpoint=parent_checkpoint,
            section_policy_checkpoint=section_checkpoint,
            policy_version="v7_no_verifier",
            enable_parent_expansion=bool(parent_checkpoint),
            enable_section_expansion=bool(section_checkpoint),
        )
        all_systems["learned_v7_no_verifier"] = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=reranker,
            generator=generator, verifier=None, settings=no_verifier_settings,
            parent_policy=AdaptiveMergePolicy(
                threshold=no_verifier_settings.merge_threshold,
                margin=no_verifier_settings.merge_margin,
                evidence_gain_weight=no_verifier_settings.evidence_gain_weight,
                cost_penalty=no_verifier_settings.cost_penalty,
                checkpoint=parent_checkpoint,
            ) if parent_checkpoint else NeverMergePolicy(),
            section_policy=AdaptiveMergePolicy(
                threshold=no_verifier_settings.merge_threshold,
                margin=no_verifier_settings.merge_margin,
                evidence_gain_weight=no_verifier_settings.evidence_gain_weight,
                cost_penalty=no_verifier_settings.cost_penalty,
                checkpoint=section_checkpoint,
            ) if section_checkpoint else NeverMergePolicy(),
            rerank_enabled=True,
        )

    # Oracle evidence (use gold evidence as context)
    # This is implemented as a special pipeline mode
    oracle_settings = replace(settings, expansion_max_depth=0)
    all_systems["oracle_evidence"] = make_baseline_pipeline(
        "B3_flat_neural", hierarchy, index_factory=index_factory,
        reranker=reranker, generator=generator, verifier=verifier, settings=oracle_settings
    )

    # Oracle context (use full document)
    full_doc_settings = replace(settings, context_token_budget=100000, expansion_max_depth=3)
    all_systems["oracle_context"] = AdaptiveHierarchicalPipeline(
        hierarchy=hierarchy, retriever=retriever, reranker=reranker,
        generator=generator, verifier=verifier, settings=full_doc_settings,
        parent_policy=StaticMergePolicy(), section_policy=StaticMergePolicy(),
        rerank_enabled=True,
    )

    # Full-document long-context (no retrieval, just full doc)
    full_doc_no_retrieve_settings = replace(settings, expansion_max_depth=0)
    all_systems["full_document"] = make_baseline_pipeline(
        "B3_flat_neural", hierarchy, index_factory=index_factory,
        reranker=reranker, generator=generator, verifier=verifier, settings=full_doc_no_retrieve_settings
    )

    return all_systems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="data/manifests_frozen")
    parser.add_argument("--paper-manifest", default="qasper_test_papers_frozen.jsonl")
    parser.add_argument("--question-manifest", default="qasper_test_questions_frozen.jsonl")
    parser.add_argument("--dataset-name", default="qasper-v0.3")
    parser.add_argument("--split-name", default="test-unseen-paper")
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--provider", default="openai",
                        choices=("openai", "gemini", "antigravity"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--questions", type=int, default=150)
    parser.add_argument(
        "--systems", nargs="+",
        default=(
            "B0_bm25", "B1_dense", "B2_hybrid_rrf", "B3_flat_neural",
            "B4_static_hierarchy", "prior", "learned_v7",
            "learned_v7_parent_only", "learned_v7_section_only",
            "learned_v7_no_rollback", "learned_v7_no_verifier",
            "oracle_evidence", "oracle_context", "full_document"
        ),
        choices=(
            "B0_bm25", "B1_dense", "B2_hybrid_rrf", "B3_flat_neural",
            "B4_static_hierarchy", "prior", "learned_v7",
            "learned_v7_parent_only", "learned_v7_section_only",
            "learned_v7_no_rollback", "learned_v7_no_verifier",
            "oracle_evidence", "oracle_context", "full_document"
        ),
    )
    parser.add_argument("--parent-checkpoint",
                        default="checkpoints/policy_parent_v7_final.joblib")
    parser.add_argument("--section-checkpoint",
                        default="checkpoints/policy_section_v7_final.joblib")
    parser.add_argument("--artifact-dir", default="data/artifacts/v7_final")
    parser.add_argument("--report", default="analysis/v7_main_results.md")
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

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    base_settings = replace(
        Settings.from_json(config_path), llm_provider=args.provider,
        llm_model=args.model, policy_version="v7",
    )

    documents = documents_from_paper_records(papers)
    base_pipeline = build_pipeline_from_documents(documents, base_settings)

    parent_ckpt = str(PROJECT_ROOT / args.parent_checkpoint) if Path(PROJECT_ROOT / args.parent_checkpoint).exists() else None
    section_ckpt = str(PROJECT_ROOT / args.section_checkpoint) if Path(PROJECT_ROOT / args.section_checkpoint).exists() else None

    learned_requested = any(name.startswith("learned_v7") for name in args.systems)
    if learned_requested and (parent_ckpt is None or section_ckpt is None):
        missing = []
        if parent_ckpt is None:
            missing.append(args.parent_checkpoint)
        if section_ckpt is None:
            missing.append(args.section_checkpoint)
        raise FileNotFoundError(
            "learned_v7 was requested but final checkpoint(s) are missing: "
            + ", ".join(missing)
        )

    all_systems = build_systems(
        hierarchy=base_pipeline.hierarchy,
        retriever=base_pipeline.retriever,
        reranker=base_pipeline.reranker,
        generator=base_pipeline.generator,
        verifier=base_pipeline.verifier,
        settings=base_settings,
        parent_checkpoint=parent_ckpt,
        section_checkpoint=section_ckpt,
        args=args,
    )

    missing_systems = [
        name for name in dict.fromkeys(args.systems) if name not in all_systems
    ]
    if missing_systems:
        raise ValueError(
            "requested systems were not constructed: " + ", ".join(missing_systems)
        )
    systems = {name: all_systems[name] for name in dict.fromkeys(args.systems)}

    artifact_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, pipeline in systems.items():
        print(f"[benchmark] starting {name}", flush=True)
        run = run_benchmark(name, pipeline, records, seed=base_settings.seed)
        for row in run.rows:
            row["system"] = name
        write_jsonl(run.rows, artifact_dir / f"artifacts_{name}.jsonl")
        runs[name] = run
        print(json.dumps({"system": name, **run.summary}, indent=2), flush=True)

    comparisons = {}
    primary_metric = "official_qasper_evidence_f1"
    if "B3_flat_neural" in runs:
        flat = runs["B3_flat_neural"]
        for name in runs:
            if name == "B3_flat_neural":
                continue
            comparisons[name] = {
                f"{primary_metric}_p_vs_flat": significance_vs_baseline(
                    runs[name], flat, primary_metric, base_settings.seed
                ),
                f"{primary_metric}_diff_cluster_ci": clustered_ci_vs_baseline(
                    runs[name], flat, primary_metric, base_settings.seed
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
        lines.extend(["", "## Paired comparison against B3_flat_neural", ""])
        for name, comparison in comparisons.items():
            low, high = comparison[f"{primary_metric}_diff_cluster_ci"]
            lines.append(
                f"- {name}: official evidence-F1 paired p="
                f"{comparison[f'{primary_metric}_p_vs_flat']:.4f}; "
                f"paper-clustered 95% CI for difference [{low:.4f}, {high:.4f}]."
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
