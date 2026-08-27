from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.schemas import (
    Claim,
    ContextBlock,
    Generation,
    Level,
    ScientificDocument,
    DocumentSection,
)
from edahr.verification import verify_generation


class KeywordVerifier:
    """Supports a claim only if the passage contains the marker word."""

    def __init__(self, marker="gold", score=0.9):
        self.marker = marker
        self.score = score

    def support_score(self, claim, evidence):
        return self.score if self.marker in evidence.lower() else 0.1


def _hierarchy():
    settings = Settings(
        child_target_tokens=10,
        child_overlap_sentences=0,
        children_per_parent=2,
        parent_overlap_children=0,
        min_child_hits=2,
    )
    document = ScientificDocument(
        document_id="doc",
        source="doc.pdf",
        sections=(DocumentSection(
            "Results",
            "Gold finding supports the claim here. Filler sentence without the marker. "
            "Another gold observation confirms it. Unrelated filler text again.",
        ),),
    )
    return HierarchyBuilder(settings).build([document]), settings


def _block(hierarchy, node_id, context_id):
    node = hierarchy.node(node_id)
    return ContextBlock(
        context_id=context_id,
        node_id=node.node_id,
        level=node.level,
        text=node.text,
        source=node.source,
        page_start=node.page_start,
        page_end=node.page_end,
        evidence_ids=node.evidence_child_ids or (node.node_id,),
        utility=0.8,
        token_count=node.token_count,
    )


class ChildLevelVerificationTests(unittest.TestCase):
    def setUp(self):
        self.hierarchy, self.settings = _hierarchy()
        self.parent_ids = [
            node.node_id
            for node in self.hierarchy.nodes.values()
            if node.level is Level.PARENT
        ]
        self.parent_id = self.parent_ids[0]

    def test_only_supporting_children_become_evidence(self):
        generation = Generation(True, (Claim("The gold findings hold.", ("C1",), 0.9),))
        block = _block(self.hierarchy, self.parent_id, "C1")
        verified, evidence, metrics = verify_generation(
            generation, [block], self.hierarchy, KeywordVerifier(), self.settings
        )
        child_quotes = [
            self.hierarchy.node(child).text.lower() for child in block.evidence_ids
        ]
        supporting = [q for q in child_quotes if "gold" in q]
        non_supporting = [q for q in child_quotes if "gold" not in q]
        self.assertTrue(supporting and non_supporting)  # mixed parent
        # Leaf selector: only the top-1 supporting child becomes a citation,
        # and the tie is flagged as ambiguous.
        self.assertEqual(len(evidence), 1)
        for item in evidence.values():
            self.assertIn("gold", item.quote.lower())
            self.assertGreaterEqual(item.support_score, self.settings.nli_support_threshold)
        # Citations were rewritten from context id to verified evidence ids.
        claim = verified.claims[0]
        self.assertNotEqual(claim.citations, ("C1",))
        self.assertTrue(set(claim.citations).issubset(set(evidence)))
        self.assertGreater(metrics["child_nli_calls"], 0)

    def test_selector_keeps_top1_and_flags_ambiguity(self):
        hierarchy, settings = _hierarchy()
        parent_ids = [
            node.node_id
            for node in hierarchy.nodes.values()
            if node.level is Level.PARENT
        ]
        blocks = [_block(hierarchy, pid, f"C{i + 1}") for i, pid in enumerate(parent_ids)]

        class GradedVerifier:
            def support_score(self, claim, evidence):
                text = evidence.lower()
                if "gold" not in text:
                    return 0.05
                return 0.9 if "finding" in text else 0.8

        generation = Generation(True, (
            Claim("The gold findings hold.", tuple(b.context_id for b in blocks), 0.9),
        ))
        verified, evidence, metrics = verify_generation(
            generation, blocks, hierarchy, GradedVerifier(), settings
        )
        self.assertEqual(len(evidence), 1)          # top-1 only
        top = next(iter(evidence.values()))
        self.assertIn("finding", top.quote.lower())  # and it is the best one
        self.assertEqual(metrics["ambiguous_claims"], 0.0)

        from dataclasses import replace as data_replace

        tight = data_replace(settings, evidence_margin=0.2)
        _, evidence2, metrics2 = verify_generation(
            generation, blocks, hierarchy, GradedVerifier(), tight
        )
        self.assertEqual(len(evidence2), 1)          # still top-1 ...
        self.assertEqual(metrics2["ambiguous_claims"], 1.0)  # ... but flagged

    def test_sibling_guard_uses_retrieved_leaf_set(self):
        from dataclasses import replace as data_replace

        child = self.hierarchy.child_ids[0]
        block = _block(self.hierarchy, child, "C1")
        generation = Generation(True, (Claim("weak support", ("C1",), 0.9),))

        class WeakVerifier:
            def support_score(self, claim, evidence):
                return 0.30

        guarded = data_replace(
            self.settings,
            nli_support_threshold=0.25,
            sibling_threshold_delta=0.10,
            lexical_support_min_coverage=0.0,
        )
        verified, evidence, metrics = verify_generation(
            generation, [block], self.hierarchy, WeakVerifier(), guarded,
            retrieved_ids={"a-different-leaf"},
        )
        self.assertFalse(verified.answerable)
        self.assertEqual(evidence, {})
        self.assertEqual(metrics["sibling_filtered_children"], 1.0)

        unguarded = data_replace(guarded, sibling_threshold_delta=0.0)
        verified2, evidence2, _ = verify_generation(
            generation, [block], self.hierarchy, WeakVerifier(), unguarded,
            retrieved_ids={"a-different-leaf"},
        )
        self.assertTrue(verified2.answerable)
        self.assertEqual(len(evidence2), 1)

    def test_claim_rejected_when_no_child_supports(self):
        generation = Generation(True, (Claim("A claim nothing supports.", ("C1",), 0.9),))
        children = [
            cid for cid in self.hierarchy.child_ids
            if "gold" not in self.hierarchy.node(cid).text.lower()
        ][:2]
        self.assertEqual(len(children), 2)
        blocks = [_block(self.hierarchy, cid, f"C{i + 1}") for i, cid in enumerate(children)]
        verified, evidence, metrics = verify_generation(
            generation, blocks, self.hierarchy, KeywordVerifier(), self.settings
        )
        self.assertFalse(verified.answerable)
        self.assertEqual(verified.claims, ())
        self.assertEqual(evidence, {})
        self.assertEqual(metrics["claims_rejected_no_child_support"], 1.0)

    def test_trace_records_lexical_fallback_as_effective_support(self):
        from dataclasses import replace as data_replace

        child = next(
            child_id for child_id in self.hierarchy.child_ids
            if "gold finding supports the claim" in self.hierarchy.node(child_id).text.lower()
        )
        block = _block(self.hierarchy, child, "C1")
        generation = Generation(True, (
            Claim("Gold finding supports the claim here.", ("C1",), 0.9),
        ))

        class NeutralVerifier:
            def support_score(self, claim, evidence):
                return 0.05

        settings = data_replace(
            self.settings,
            nli_support_threshold=0.6,
            lexical_support_min_coverage=0.8,
            sibling_threshold_delta=0.0,
        )
        supports: list[tuple[str, float]] = []
        trace: list[dict] = []
        verified, evidence, metrics = verify_generation(
            generation, [block], self.hierarchy, NeutralVerifier(), settings,
            claim_supports=supports, retrieved_ids={child},
            verification_trace=trace,
        )

        self.assertTrue(verified.answerable)
        self.assertEqual(len(evidence), 1)
        self.assertGreaterEqual(supports[0][1], 0.8)
        self.assertEqual(trace[0]["status"], "accepted")
        candidate = trace[0]["candidates"][0]
        self.assertAlmostEqual(candidate["nli_support"], 0.05)
        self.assertGreaterEqual(candidate["effective_support"], 0.8)
        self.assertTrue(candidate["selected"])
        self.assertEqual(metrics["claims_rejected_base_support"], 0.0)


if __name__ == "__main__":
    unittest.main()
