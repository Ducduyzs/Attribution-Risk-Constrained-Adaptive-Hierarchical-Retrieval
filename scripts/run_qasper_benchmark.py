"""Run the frozen benchmark on paper/question manifests with full baseline/ablation support."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from typing import Callable
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import (  # noqa: E402
    auto_label_gold_children,
    raptor_faithful_retriever,
    clustered_ci_vs_baseline,
    make_baseline_pipeline,
    run_benchmark,
    significance_vs_baseline,
)
from edahr.config import Settings  # noqa: E402
from edahr.context_baselines import FullDocumentPipeline, OracleEvidencePipeline  # noqa: E402
from edahr.pipeline import AdaptiveHierarchicalPipeline  # noqa: E402
from edahr.policy import AdaptiveMergePolicy, NeverMergePolicy  # noqa: E402
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


def release_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def value(summary: dict, key: str) -> float:
    raw = summary.get(key)
    return float(raw) if isinstance(raw, (int, float)) else 0.0


SYSTEM_NAMES = (
    "B0_bm25", "B1_dense", "B2_hybrid_rrf", "B3_flat_neural",
    "B4_static_hierarchy", "B5_raptor_faithful", "B6_longrag_faithful",
    "prior", "prior_raptor", "learned_v7",
    "learned_v7_parent_only", "learned_v7_section_only",
    "learned_v7_no_rollback", "learned_v7_no_verifier",
    "oracle_evidence", "full_document",
)


def gold_children_by_question(hierarchy, records: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Gold evidence leaves per (source, query), as labelled for every metric."""
    gold: dict[tuple[str, str], list[str]] = {}
    for record in records:
        key = (str(record.get("source") or ""), record["query"])
        if key in gold:
            raise ValueError(f"duplicate (source, query) in manifest: {key!r}")
        children, _ = auto_label_gold_children(hierarchy, record)
        gold[key] = sorted(children)
    return gold


def build_systems(
    hierarchy,
    retriever,
    reranker,
    generator,
    verifier,
    settings: Settings,
    parent_checkpoint: str | None,
    section_checkpoint: str | None,
    records: list[dict],
    full_document_token_budget: int,
) -> dict[str, Callable[[], AdaptiveHierarchicalPipeline]]:
    """Lazy factories sharing heavy components.

    Each system is constructed only when it runs, so baselines that build
    their own index never coexist in RAM/VRAM.
    """
    encoder = getattr(retriever, "encoder", None)

    def index_factory(s: Settings):
        from edahr.index import MultiRepresentationIndex
        return MultiRepresentationIndex(hierarchy, encoder, s)

    def baseline(name: str) -> Callable[[], AdaptiveHierarchicalPipeline]:
        return lambda: make_baseline_pipeline(
            name, hierarchy, index_factory=index_factory,
            reranker=reranker, generator=generator, verifier=verifier, settings=settings,
        )

    def gate(s: Settings, checkpoint: str | None):
        if not checkpoint:
            return NeverMergePolicy()
        return AdaptiveMergePolicy(
            threshold=s.merge_threshold, margin=s.merge_margin,
            evidence_gain_weight=s.evidence_gain_weight,
            cost_penalty=s.cost_penalty, checkpoint=checkpoint,
        )

    def learned(version: str, parent: str | None, section: str | None,
                use_verifier: bool = True, **overrides) -> Callable[[], AdaptiveHierarchicalPipeline]:
        def make() -> AdaptiveHierarchicalPipeline:
            s = replace(
                settings,
                parent_policy_checkpoint=parent,
                section_policy_checkpoint=section,
                policy_version=version,
                enable_parent_expansion=bool(parent),
                enable_section_expansion=bool(section),
                **overrides,
            )
            return AdaptiveHierarchicalPipeline(
                hierarchy=hierarchy, retriever=retriever, reranker=reranker,
                generator=generator, verifier=verifier if use_verifier else None,
                settings=s, parent_policy=gate(s, parent),
                section_policy=gate(s, section), rerank_enabled=True,
            )
        return make

    def prior(retriever_factory=lambda s: retriever) -> AdaptiveHierarchicalPipeline:
        s = replace(
            settings, parent_policy_checkpoint=None, section_policy_checkpoint=None,
            policy_version="prior", enable_parent_expansion=True,
            enable_section_expansion=True,
        )
        policy = AdaptiveMergePolicy(
            threshold=s.merge_threshold, margin=s.merge_margin,
            evidence_gain_weight=s.evidence_gain_weight, cost_penalty=s.cost_penalty,
        )
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever_factory(s), reranker=reranker,
            generator=generator, verifier=verifier, settings=s,
            parent_policy=policy, section_policy=policy, rerank_enabled=True,
        )

    factories: dict[str, Callable[[], AdaptiveHierarchicalPipeline]] = {
        name: baseline(name)
        for name in ("B0_bm25", "B1_dense", "B2_hybrid_rrf",
                     "B3_flat_neural", "B4_static_hierarchy",
                     "B5_raptor_faithful", "B6_longrag_faithful")
    }
    factories["prior"] = prior
    # Prior adaptive expansion + verification on RAPTOR's collapsed-tree retrieval.
    factories["prior_raptor"] = lambda: prior(
        lambda s: raptor_faithful_retriever(hierarchy, s))

    if parent_checkpoint or section_checkpoint:
        factories["learned_v7"] = learned("v7", parent_checkpoint, section_checkpoint)
        # Inference-time rollback disabled. This is NOT the same experiment as
        # removing the drift term from the training reward.
        factories["learned_v7_no_rollback"] = learned(
            "v7_no_rollback", parent_checkpoint, section_checkpoint, rollback_ratio=0.0,
        )
        factories["learned_v7_no_verifier"] = learned(
            "v7_no_verifier", parent_checkpoint, section_checkpoint, use_verifier=False,
        )
    if parent_checkpoint:
        factories["learned_v7_parent_only"] = learned("v7-parent_only", parent_checkpoint, None)
    if section_checkpoint:
        factories["learned_v7_section_only"] = learned("v7_section_only", None, section_checkpoint)

    # Upper bound: the gold evidence leaves themselves are the context.
    factories["oracle_evidence"] = lambda: OracleEvidencePipeline(
        hierarchy=hierarchy, retriever=retriever, reranker=reranker,
        generator=generator, verifier=verifier, settings=settings,
        gold_children=gold_children_by_question(hierarchy, records),
    )
    # Long-context baseline: the whole paper, no retrieval.
    factories["full_document"] = lambda: FullDocumentPipeline(
        hierarchy=hierarchy, retriever=retriever, reranker=reranker,
        generator=generator, verifier=verifier, settings=settings,
        token_budget=full_document_token_budget,
    )
    return factories


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
        "--systems", nargs="+", default=SYSTEM_NAMES, choices=SYSTEM_NAMES,
    )
    parser.add_argument(
        "--full-document-token-budget", type=int, default=100_000,
        help="context budget for full_document (must fit the generator window)",
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
        records=records,
        full_document_token_budget=args.full_document_token_budget,
    )

    missing_systems = [
        name for name in dict.fromkeys(args.systems) if name not in all_systems
    ]
    if missing_systems:
        raise ValueError(
            "requested systems were not constructed: " + ", ".join(missing_systems)
        )
    factories = {name: all_systems[name] for name in dict.fromkeys(args.systems)}

    artifact_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, factory in factories.items():
        print(f"[benchmark] starting {name}", flush=True)
        pipeline = factory()
        run = run_benchmark(name, pipeline, records, seed=base_settings.seed)
        del pipeline
        release_memory()
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
