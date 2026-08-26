"""End-to-end smoke test with lightweight fakes (no GPU models needed).

Exercises: retrieval -> rerank -> adaptive parent merge -> iterative expansion
to section/document -> guarded context assembly -> child-level verification,
then prints the key metrics a benchmark run would consume.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.context import assemble_context
from edahr.expansion import expand_selection
from edahr.hierarchy import HierarchyBuilder
from edahr.pipeline import AdaptiveHierarchicalPipeline, classify_query
from edahr.policy import AdaptiveMergePolicy, decide_merges
from edahr.schemas import Claim, DocumentSection, Generation, Hit, Level, ScientificDocument
from edahr.verification import verify_generation


class Retriever:
    def __init__(self, ids): self.ids = ids
    def search(self, q, k):
        return [Hit(node_id=nid, score=0.9 - i * 0.02, rank=i + 1)
                for i, nid in enumerate(self.ids[:k])]


class Reranker:
    """Long consolidated texts score higher (stands in for a cross-encoder)."""
    def score(self, q, texts):
        return [0.95 if len(t) > 120 else 0.55 for t in texts]


class Generator:
    def generate(self, q, ctx):
        # Realistic multi-citation: one drifting citation (Analysis parent)
        # plus the genuinely supportive block; child-level NLI must keep only
        # evidence whose own passage supports the claim.
        return Generation(True, (
            Claim("The model merges parents when utility improves.",
                  tuple(block.context_id for block in ctx[:2]), 0.88),
        ))


class Verifier:
    def support_score(self, claim, passage):
        return 0.91 if "utility" in passage.lower() else 0.20


def main():
    settings = Settings(
        child_target_tokens=14, children_per_parent=2, parent_overlap_children=0,
        candidate_k=12, rerank_k=12, final_context_k=5, context_token_budget=260,
        min_child_hits=2, merge_margin=-0.05, rollback_ratio=0.5,
        expansion_max_depth=3, knapsack_token_bucket=16,
    )
    docs = [
        ScientificDocument("paperA", "a.pdf", (
            DocumentSection("Method", "Utility drives parent merging decisions here. "
                "Margin guards against low value consolidation. Children stay when "
                "specificity matters most for factoid style queries."),
            DocumentSection("Results", "Results show density aware merging wins. "
                "Gains hold across multiple scientific benchmarks too. Latency stays "
                "within the strict token budget envelope overall."),
        )),
        ScientificDocument("paperB", "b.pdf", (
            DocumentSection("Analysis", "Analysis confirms the adaptive controller. "
                "Evidence density predicts answer quality well. Provenance remains at "
                "child granularity throughout every stage."),
        )),
    ]
    hierarchy = HierarchyBuilder(settings).build(docs)

    query = "Summarize the overall findings about utility and merging"
    query_type = classify_query(query)
    hits = Retriever(list(hierarchy.child_ids)).search(query, settings.candidate_k)
    scores = Reranker().score(query, [hierarchy.node(h.node_id).text for h in hits])
    from dataclasses import replace
    reranked = sorted(
        [replace(h, reranker_score=s) for h, s in zip(hits, scores)],
        key=lambda h: h.reranker_score, reverse=True)
    cache = {h.node_id: h.reranker_score for h in reranked}
    members = dict(cache)

    policy = AdaptiveMergePolicy(margin=settings.merge_margin)
    parent_decisions = decide_merges(query, query_type, hierarchy, reranked,
                                     Reranker(), policy, settings,
                                     candidate_score_cache=cache)
    selected = dict(members)
    for d in parent_decisions:
        if d.accepted:
            for m in d.child_ids:
                selected.pop(m, None)
            selected[d.parent_id] = cache.get(d.parent_id, d.parent_utility)

    expanded, exp_decisions, trace = expand_selection(
        query, query_type, hierarchy, Reranker(), policy, settings,
        selected, score_cache=cache, start_rank=int(Level.PARENT is not None) * 1)

    blocks = assemble_context(hierarchy, expanded, query_type, settings)
    gen = Generator().generate(query, blocks)
    verified, evidence, vmetrics = verify_generation(gen, blocks, hierarchy,
                                                     Verifier(), settings)

    print("query_type:", query_type.value)
    print("trace:", trace)
    all_decisions = list(parent_decisions) + list(exp_decisions)
    print("decisions:", [(d.level.value, d.accepted, round(d.utility, 3))
                         for d in all_decisions])
    print("context tokens:", sum(b.token_count for b in blocks),
          "<= budget", settings.context_token_budget)
    print("levels in context:", {b.level.value for b in blocks})
    print("verified claims:", len(verified.claims),
          "| citations->evidence:", verified.claims[0].citations if verified.claims else ())
    print("evidence pages:", sorted({(e.source, e.page_start) for e in evidence.values()}))
    assert sum(b.token_count for b in blocks) <= settings.context_token_budget
    assert any(d.level in (Level.SECTION, Level.DOCUMENT) and d.accepted
               for d in exp_decisions), "global query should expand beyond parent"
    # Citation drift guard: the drifting Analysis citation must not leak
    # unsupported children into evidence; every kept quote mentions utility.
    assert evidence, "supportive block's children should pass child-level NLI"
    for ev in evidence.values():
        assert "utility" in ev.quote.lower(), "only supportive children may remain"
        assert ev.page_start >= 1
        assert ev.char_end >= ev.char_start >= 0
    print("SMOKE OK")


if __name__ == "__main__":
    main()
