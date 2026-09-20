"""Offline full-ladder benchmark incl. RAPTOR (B5) + LongRAG (B6) baselines.

Runs every system in ``BASELINE_NAMES`` over real QASPER dev papers with
dependency-free backends (BM25 retrieval, lexical rerank, extractive
generation, token-overlap verification), so the comparison executes the
real pipeline code paths without model downloads or LLM keys.

Outputs:
  analysis/b5b6_lexical_ladder.json  -- per-system summaries + paired stats
  analysis/b5b6_lexical_ladder.md    -- human-readable report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import (  # noqa: E402
    BASELINE_NAMES,
    Bm25ChildRetriever,
    clustered_ci_vs_baseline,
    make_baseline_pipeline,
    run_benchmark,
    significance_vs_baseline,
)
from edahr.config import Settings  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import convert_qasper, documents_from_paper_records  # noqa: E402
from edahr.schemas import Claim, Generation  # noqa: E402


class LexicalReranker:
    """Query-child token-overlap scorer (deterministic, dependency-free)."""

    def score(self, query: str, texts) -> list[float]:
        query_tokens = Counter(query.lower().split())
        scores: list[float] = []
        for text in texts:
            text_tokens = Counter(str(text).lower().split())
            common = sum(min(c, text_tokens.get(t, 0)) for t, c in query_tokens.items())
            scores.append(common / max(1, sum(query_tokens.values())))
        return scores


class ExtractiveGenerator:
    """Emit claims as leading sentences of top context blocks (with citations)."""

    def __init__(self, max_claims: int = 3):
        self.max_claims = max_claims

    def generate(self, query: str, context):
        claims: list[Claim] = []
        for block in list(context)[: self.max_claims]:
            sentences = re.split(r"(?<=[.!?])\s+", block.text.strip())
            sentence = next((s.strip() for s in sentences if s.strip()), "")
            if sentence:
                claims.append(Claim(sentence, (block.context_id,), 0.8))
        return Generation(bool(claims), tuple(claims))


class OverlapVerifier:
    """Token-F1 support score between claim and candidate evidence."""

    def support_score(self, claim: str, evidence: str) -> float:
        claim_tokens = claim.lower().split()
        evidence_tokens = evidence.lower().split()
        if not claim_tokens or not evidence_tokens:
            return 0.0
        evidence_counts = Counter(evidence_tokens)
        common = sum(
            min(c, evidence_counts.get(t, 0)) for t, c in Counter(claim_tokens).items()
        )
        if not common:
            return 0.0
        precision = common / len(claim_tokens)
        recall = common / len(evidence_tokens)
        return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=int, default=10)
    parser.add_argument("--max-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    papers, questions, report = convert_qasper(
        PROJECT_ROOT / "data" / "qasper" / "qasper-dev-v0.3.json", "dev"
    )
    eligible = [q for q in questions if q.get("gold_quotes")]
    selected_questions = eligible[: args.max_questions]
    used_papers = {str(q["paper_id"]) for q in selected_questions}
    selected_papers = [p for p in papers if str(p["paper_id"]) in used_papers][
        : args.papers
    ]
    # Keep only questions whose paper survived the paper cap.
    kept_ids = {str(p["paper_id"]) for p in selected_papers}
    selected_questions = [q for q in selected_questions if str(q["paper_id"]) in kept_ids]

    settings = Settings(seed=args.seed)
    documents = documents_from_paper_records(selected_papers)
    hierarchy = HierarchyBuilder(settings).build(documents)
    print(f"papers={len(selected_papers)} questions={len(selected_questions)} "
          f"children={len(hierarchy.child_ids)}")

    def index_factory(variant: Settings):
        return Bm25ChildRetriever(hierarchy, variant.bm25_k1, variant.bm25_b)

    reranker = LexicalReranker()
    generator = ExtractiveGenerator()
    verifier = OverlapVerifier()

    runs = {}
    for name in BASELINE_NAMES:
        pipeline = make_baseline_pipeline(
            name, hierarchy, index_factory=index_factory, reranker=reranker,
            generator=generator, verifier=verifier, settings=settings,
        )
        run = run_benchmark(name, pipeline, selected_questions, seed=args.seed)
        runs[name] = run
        summary = run.summary
        print(
            f"{name:22s} recall@5={summary.get('recall@5', 0):.4f} "
            f"citF1={summary.get('citation_f1', 0):.4f} "
            f"ansF1={summary.get('answer_f1', 0):.4f} "
            f"tok={summary.get('context_tokens', 0):.1f}"
        )

    baseline = runs["B3_flat_neural"]
    comparison = {}
    for name, run in runs.items():
        if name == "B3_flat_neural":
            continue
        p_value = significance_vs_baseline(run, baseline, "citation_f1", seed=args.seed)
        ci_low, ci_high = clustered_ci_vs_baseline(
            run, baseline, "citation_f1", seed=args.seed
        )
        comparison[name] = {
            "paired_p_vs_B3": p_value,
            "clustered_ci_low_vs_B3": ci_low,
            "clustered_ci_high_vs_B3": ci_high,
        }

    payload = {
        "papers": len(selected_papers),
        "questions": len(selected_questions),
        "qasper_report": report,
        "backend": "offline-lexical (bm25/overlap/extractive); no neural weights, no LLM key",
        "summaries": {name: run.summary for name, run in runs.items()},
        "comparison_vs_B3_flat_neural": comparison,
    }
    out_json = PROJECT_ROOT / "analysis" / "b5b6_lexical_ladder.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Offline full-ladder benchmark (B0-B6 + edahr) — QASPER dev subset",
        "",
        f"Papers/questions: {len(selected_papers)}/{len(selected_questions)}. "
        "Backend: BM25 retrieval, lexical rerank, extractive generation, "
        "token-overlap verification (no neural weights, no LLM key). "
        "All systems share ingestion/hierarchy/generation/verification; only "
        "retrieval granularity and hierarchy control differ.",
        "",
        "| system | recall@5 | citation F1 | answer F1 | ctx tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in BASELINE_NAMES:
        summary = runs[name].summary
        lines.append(
            f"| {name} | {summary.get('recall@5', 0):.4f} | "
            f"{summary.get('citation_f1', 0):.4f} | {summary.get('answer_f1', 0):.4f} | "
            f"{summary.get('context_tokens', 0):.1f} |"
        )
    lines += [
        "",
        "## Paired comparison vs B3_flat_neural (citation F1)",
        "",
    ]
    for name, stats in comparison.items():
        lines.append(
            f"- {name}: paired p={stats['paired_p_vs_B3']:.4f}; "
            f"paper-clustered 95% CI "
            f"[{stats['clustered_ci_low_vs_B3']:.4f}, {stats['clustered_ci_high_vs_B3']:.4f}]"
        )
    out_md = PROJECT_ROOT / "analysis" / "b5b6_lexical_ladder.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
