from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.pipeline import AdaptiveHierarchicalPipeline, classify_query
from edahr.policy import AdaptiveMergePolicy
from edahr.schemas import (
    Claim,
    DocumentSection,
    Generation,
    Hit,
    Level,
    QueryType,
    ScientificDocument,
)


class FakeRetriever:
    def __init__(self, node_ids):
        self.node_ids = node_ids

    def search(self, query, k):
        return [Hit(node_id=value, score=0.9 - index * 0.03, rank=index + 1)
                for index, value in enumerate(self.node_ids[:k])]


class FakeReranker:
    def score(self, query, texts):
        return [0.92 - index * 0.01 for index, _ in enumerate(texts)]


class FakeGenerator:
    def generate(self, query, context):
        return Generation(True, (Claim("Adaptive retrieval improves evidence selection.", ("C1",), 0.9),))


class SelectiveVerifier:
    """Child-level verifier: only passages mentioning 'adaptive' entail the claim."""

    calls = 0

    def support_score(self, claim, evidence):
        SelectiveVerifier.calls += 1
        return 0.95 if "adaptive" in evidence.lower() else 0.05


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            child_target_tokens=8,
            child_overlap_sentences=0,
            children_per_parent=3,
            parent_overlap_children=0,
            candidate_k=6,
            rerank_k=6,
            final_context_k=3,
            merge_threshold=0.0,
            merge_margin=0.0,
            rollback_ratio=0.5,
            min_child_hits=2,
        )
        document = ScientificDocument(
            document_id="paper-1",
            source="paper.pdf",
            sections=(DocumentSection(
                "Methods",
                "Adaptive retrieval selects evidence. Dense retrieval finds semantic matches. "
                "Sparse retrieval preserves terminology. Late interaction aligns individual tokens. "
                "A reranker improves final ranking. NLI verifies generated scientific claims.",
                section_type="methods",
            ),),
        )
        self.hierarchy = HierarchyBuilder(self.settings).build([document])
        SelectiveVerifier.calls = 0

    def test_query_classifier(self):
        self.assertEqual(classify_query("How does the system work?"), QueryType.EXPLANATORY)
        self.assertEqual(classify_query("Compare dense and sparse retrieval"), QueryType.COMPARATIVE)

    def test_end_to_end_with_grounded_claim(self):
        pipeline = AdaptiveHierarchicalPipeline(
            hierarchy=self.hierarchy,
            retriever=FakeRetriever(list(self.hierarchy.child_ids)),
            reranker=FakeReranker(),
            generator=FakeGenerator(),
            verifier=SelectiveVerifier(),
            settings=self.settings,
            policy=AdaptiveMergePolicy(threshold=0.0, margin=0.0),
        )
        result = pipeline.answer("How does adaptive retrieval work?")
        self.assertTrue(result.generation.answerable)
        self.assertEqual(len(result.generation.claims), 1)
        self.assertGreaterEqual(len(result.evidence), 1)
        self.assertGreaterEqual(len(result.decisions), 1)
        self.assertEqual(result.metrics["verified_claims"], 1.0)
        # Child-level verification: every kept evidence passage must itself
        # support the claim (no blanket descendant inheritance).
        for evidence in result.evidence.values():
            self.assertIn("adaptive", evidence.quote.lower())
        # Verified citations point at evidence ids, not raw context ids.
        claim = result.generation.claims[0]
        self.assertTrue(claim.citations)
        for citation in claim.citations:
            self.assertIn(citation, result.evidence)
        # Latency instrumentation present.
        self.assertIn("total_latency_ms", result.metrics)
        self.assertIn("child_nli_calls", result.metrics)

    def test_legacy_retriever_is_scoped_before_top_k(self):
        settings = Settings(
            child_target_tokens=8,
            child_overlap_sentences=0,
            children_per_parent=2,
            parent_overlap_children=0,
            candidate_k=1,
            rerank_k=1,
            final_context_k=1,
            min_child_hits=2,
            expansion_max_depth=0,
        )
        documents = [
            ScientificDocument(
                document_id="foreign",
                source="foreign.pdf",
                sections=(DocumentSection("S", "Noise passage one. Noise passage two."),),
            ),
            ScientificDocument(
                document_id="target",
                source="target.pdf",
                sections=(DocumentSection(
                    "S", "Adaptive retrieval selects evidence. Target detail follows."
                ),),
            ),
        ]
        hierarchy = HierarchyBuilder(settings).build(documents)

        class LegacyRetriever:
            def __init__(self):
                self.requested = 0
                self.order = [
                    *[cid for cid in hierarchy.child_ids if hierarchy.node(cid).source == "foreign.pdf"],
                    *[cid for cid in hierarchy.child_ids if hierarchy.node(cid).source == "target.pdf"],
                ]

            def search(self, query, k):
                self.requested = k
                return [Hit(node_id=cid, score=1.0 - i * 0.01, rank=i + 1)
                        for i, cid in enumerate(self.order[:k])]

        retriever = LegacyRetriever()
        pipeline = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=FakeReranker(),
            generator=FakeGenerator(), verifier=SelectiveVerifier(), settings=settings,
        )
        result = pipeline.answer("adaptive retrieval", source="target.pdf")
        self.assertEqual(retriever.requested, len(hierarchy.child_ids))
        self.assertTrue(result.hits)
        self.assertTrue(all(hierarchy.node(hit.node_id).source == "target.pdf"
                            for hit in result.hits))


if __name__ == "__main__":
    unittest.main()
