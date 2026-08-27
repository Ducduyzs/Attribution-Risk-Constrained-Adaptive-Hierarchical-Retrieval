from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.baselines import (
    BASELINE_NAMES,
    BenchmarkRun,
    Bm25ChildRetriever,
    RrfRetriever,
    auto_label_gold_children,
    clustered_ci_vs_baseline,
    make_baseline_pipeline,
    run_benchmark,
    significance_vs_baseline,
    split_by_paper,
)
from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.schemas import Claim, DocumentSection, Generation, Hit, ScientificDocument


class LexicalRetriever:
    """Deterministic retriever: ranks children by marker count desc."""

    def search(self, query, k):
        return []


class FakeReranker:
    def score(self, query, texts):
        return [0.9 for _ in texts]


class FakeGenerator:
    def generate(self, query, context):
        claims = (
            Claim("gold finding confirmed", (context[0].context_id,), 0.9),
        )
        return Generation(True, claims)


class FakeVerifier:
    def support_score(self, claim, evidence):
        return 0.95 if "gold" in evidence.lower() else 0.1


def _hierarchy():
    settings = Settings(
        child_target_tokens=10,
        child_overlap_sentences=0,
        children_per_parent=2,
        parent_overlap_children=0,
        min_child_hits=2,
        final_context_k=3,
        context_token_budget=200,
        merge_margin=-1.0,
    )
    document = ScientificDocument(
        document_id="doc",
        source="doc.pdf",
        sections=(DocumentSection(
            "Results",
            "Gold finding alpha here now. Filler beta text without marker. "
            "Second gold gamma observation. More filler delta content.",
        ),),
    )
    return HierarchyBuilder(settings).build([document]), settings


class Bm25Tests(unittest.TestCase):
    def test_bm25_ranks_matching_children_first(self):
        hierarchy, _ = _hierarchy()
        retriever = Bm25ChildRetriever(hierarchy)
        hits = retriever.search("gold", k=len(hierarchy.child_ids))
        self.assertEqual(len(hits), len(hierarchy.child_ids))
        first_nonmatching = next(
            (i for i, hit in enumerate(hits) if hit.score <= 0.0), len(hits)
        )
        self.assertGreater(first_nonmatching, 0)
        for hit in hits[:first_nonmatching]:
            self.assertIn("gold", hierarchy.node(hit.node_id).text.lower())
        self.assertGreaterEqual(first_nonmatching, 2)  # two gold children exist
        self.assertGreater(hits[0].score, 0.0)

    def test_bm25_source_filter_applies_before_k(self):
        hierarchy, _ = _hierarchy()
        hits = Bm25ChildRetriever(hierarchy).search("gold", k=1, source="doc.pdf")
        self.assertEqual(hierarchy.node(hits[0].node_id).source, "doc.pdf")

    def test_rrf_fuses_rankings(self):
        class StaticRetriever:
            def __init__(self, order):
                self.order = order

            def search(self, query, k):
                return [
                    Hit(node_id=node_id, score=1.0 - i * 0.1, rank=i + 1)
                    for i, node_id in enumerate(self.order[:k])
                ]

        a, b, c = "n1", "n2", "n3"
        fused = RrfRetriever([StaticRetriever([a, b, c]), StaticRetriever([b, c])], rrf_k=60)
        hits = fused.search("q", k=3)
        self.assertEqual(hits[0].node_id, b)          # top in both lists
        self.assertGreater(hits[0].score, hits[2].score)


class SplitTests(unittest.TestCase):
    def test_paper_level_split_disjoint(self):
        splits = split_by_paper(["p1", "p2", "p3", "p4", "p5", "p6", "p7"], seed=7)
        union = set(splits["train"]) | set(splits["calibration"]) | set(splits["test"])
        self.assertEqual(union, {"p1", "p2", "p3", "p4", "p5", "p6", "p7"})
        for first in ("train", "calibration"):
            for second in ("test",):
                self.assertFalse(set(splits[first]) & set(splits[second]))


class GoldLabellingTests(unittest.TestCase):
    def test_quotes_map_to_children(self):
        hierarchy, _ = _hierarchy()
        record = {"query": "q", "gold_quotes": ["Gold finding alpha here now."]}
        gold_ids, quotes = auto_label_gold_children(hierarchy, record)
        self.assertTrue(gold_ids)
        self.assertEqual(len(quotes), 1)
        for child_id in gold_ids:
            self.assertIn(child_id, hierarchy.child_ids)


class BenchmarkHarnessTests(unittest.TestCase):
    def test_run_benchmark_produces_summary(self):
        hierarchy, settings = _hierarchy()
        from edahr.pipeline import AdaptiveHierarchicalPipeline
        from edahr.policy import AdaptiveMergePolicy

        pipeline = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy,
            retriever=Bm25ChildRetriever(hierarchy),
            reranker=FakeReranker(),
            generator=FakeGenerator(),
            verifier=FakeVerifier(),
            settings=settings,
            policy=AdaptiveMergePolicy(margin=-1.0),
        )
        records = [{
            "query": "What is the gold finding?",
            "gold_quotes": ["Gold finding alpha here now."],
            "answer": "gold finding confirmed",
            "gold_pages": {"doc.pdf": 1},
        }]
        run = run_benchmark("smoke", pipeline, records, ks=(1, 2))
        self.assertIn("recall@2", run.rows[0])
        self.assertIn("citation_f1", run.summary)
        self.assertEqual(run.summary["num_queries"], 1.0)
        self.assertIn("latency_p95_ms", run.summary)
        self.assertTrue(run.rows[0]["citation_evaluable"])
        self.assertIn("question_id", run.rows[0])
        self.assertEqual(run.rows[0]["generated_claim_count"], 1)
        self.assertEqual(run.rows[0]["verified_claim_count"], 1)
        self.assertEqual(
            len(run.rows[0]["evidence_node_ids"]),
            len(set(run.rows[0]["evidence_node_ids"])),
        )

    def test_unmapped_gold_is_excluded_from_citation_macro(self):
        hierarchy, settings = _hierarchy()
        from edahr.pipeline import AdaptiveHierarchicalPipeline
        pipeline = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=Bm25ChildRetriever(hierarchy),
            reranker=FakeReranker(), generator=FakeGenerator(),
            verifier=FakeVerifier(), settings=settings,
        )
        records = [{"query": "gold", "source": "doc.pdf", "gold_quotes": []}]
        run = run_benchmark("zero-gold", pipeline, records, ks=(1,))
        self.assertFalse(run.rows[0]["citation_evaluable"])
        self.assertIsNone(run.rows[0]["citation_f1"])
        self.assertEqual(run.summary["citation_evaluable_queries"], 0.0)

    def test_make_baseline_pipeline_names(self):
        hierarchy, settings = _hierarchy()

        def index_factory(variant_settings):
            return Bm25ChildRetriever(hierarchy)  # cheap stand-in for tests

        for name in BASELINE_NAMES:
            pipeline = make_baseline_pipeline(
                name,
                hierarchy,
                index_factory=index_factory,
                reranker=FakeReranker(),
                generator=FakeGenerator(),
                verifier=FakeVerifier(),
                settings=settings,
            )
            self.assertIsInstance(pipeline.hierarchy.nodes, dict)

    def test_paired_statistics_align_by_identity_and_skip_none(self):
        proposed = BenchmarkRun("proposed", rows=[
            {"source": "p2", "question_id": "q2", "citation_f1": None},
            {"source": "p1", "question_id": "q1", "citation_f1": 0.8},
            {"source": "p1", "question_id": "q3", "citation_f1": 0.6},
        ])
        baseline = BenchmarkRun("flat", rows=[
            {"source": "p1", "question_id": "q3", "citation_f1": 0.4},
            {"source": "p1", "question_id": "q1", "citation_f1": 0.5},
            {"source": "p2", "question_id": "q2", "citation_f1": None},
        ])
        p_value = significance_vs_baseline(proposed, baseline, seed=3)
        ci_low, ci_high = clustered_ci_vs_baseline(proposed, baseline, seed=3)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        self.assertAlmostEqual(ci_low, 0.25)
        self.assertAlmostEqual(ci_high, 0.25)


if __name__ == "__main__":
    unittest.main()
