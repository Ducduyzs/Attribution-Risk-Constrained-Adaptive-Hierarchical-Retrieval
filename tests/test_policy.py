from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.policy import (
    AdaptiveMergePolicy,
    decide_candidates,
    decide_merges,
)
from edahr.schemas import DocumentSection, Hit, QueryType, ScientificDocument


class MappedReranker:
    """Scores texts by marker: 'strong' -> 0.95, 'weak' -> 0.30."""

    def score(self, query, texts):
        return [0.95 if "strong" in text.lower() else 0.30 for text in texts]


def _hierarchy():
    settings = Settings(
        child_target_tokens=12,
        child_overlap_sentences=0,
        children_per_parent=3,
        parent_overlap_children=0,
        min_child_hits=2,
        context_token_budget=5000,
    )
    document = ScientificDocument(
        document_id="doc",
        source="doc.pdf",
        sections=(DocumentSection(
            "Methods",
            "Strong alpha sentence one. Strong beta sentence two. Weak gamma sentence three. "
            "Weak delta sentence four. Strong epsilon sentence five. Weak zeta sentence six.",
        ),),
    )
    return HierarchyBuilder(settings).build([document]), settings


def _child_hits(hierarchy):
    hits = [
        Hit(node_id=cid, score=0.5 + i * 0.01, rank=i + 1)
        for i, cid in enumerate(hierarchy.child_ids)
    ]
    return hits


class UtilityMarginTests(unittest.TestCase):
    def setUp(self):
        self.hierarchy, self.settings = _hierarchy()
        self.parent_ids = [
            node.node_id
            for node in self.hierarchy.nodes.values()
            if node.level.name == "PARENT"
        ]

    def test_decision_fields_populated(self):
        decisions = decide_merges(
            "query", QueryType.FACTOID, self.hierarchy, _child_hits(self.hierarchy),
            MappedReranker(), AdaptiveMergePolicy(margin=0.0), self.settings,
        )
        self.assertEqual(len(decisions), len(self.parent_ids))
        decision = decisions[0]
        self.assertGreater(decision.parent_utility, 0.0)
        self.assertGreater(decision.children_utility, 0.0)
        self.assertIsInstance(decision.cost_delta_tokens, int)
        # relevance feature is the query-parent rerank score, not child mean.
        parent_score = 0.95  # parents are built from mixed children incl. 'strong'
        self.assertAlmostEqual(decision.features.relevance, parent_score)

    def test_margin_blocks_low_value_merge(self):
        # Parent mixes strong+weak children; members include a 'strong' child
        # (0.95) while the parent itself scores 0.95 too -- with an enormous
        # margin no merge may pass.
        policy = AdaptiveMergePolicy(margin=10.0)
        decisions = decide_merges(
            "q", QueryType.FACTOID, self.hierarchy, _child_hits(self.hierarchy),
            MappedReranker(), policy, self.settings,
        )
        self.assertTrue(decisions)
        self.assertTrue(all(not d.accepted for d in decisions))

    def test_rollback_guard(self):
        class WeakParentReranker(MappedReranker):
            """Parents are long joins of children; make them all score low."""

            def score(self, query, texts):
                values = []
                for text in texts:
                    if len(text) > 60:
                        values.append(0.10)
                    else:
                        values.append(0.95 if "strong" in text.lower() else 0.60)
                return values

        settings_with_guard = Settings(
            rollback_ratio=0.78, min_child_hits=2, context_token_budget=5000,
        )
        decisions = decide_merges(
            "q", QueryType.FACTOID, self.hierarchy, _child_hits(self.hierarchy),
            WeakParentReranker(), AdaptiveMergePolicy(margin=-1.0), settings_with_guard,
        )
        self.assertTrue(any(d.rolled_back and not d.accepted for d in decisions))

    def test_generic_candidates_over_levels(self):
        from edahr.schemas import Level

        section_ids = [
            node.node_id
            for node in self.hierarchy.nodes.values()
            if node.level is Level.SECTION
        ]
        member_scores = {cid: 0.4 for cid in self.hierarchy.child_ids}
        decisions = decide_candidates(
            query="q",
            query_type=QueryType.GLOBAL,
            hierarchy=self.hierarchy,
            reranker=MappedReranker(),
            policy=AdaptiveMergePolicy(margin=0.0),
            settings=self.settings,
            candidates=section_ids,
            member_scores=member_scores,
            candidate_level=Level.SECTION,
            member_level=Level.CHILD,
        )
        self.assertEqual(len(decisions), len(section_ids))
        self.assertEqual(decisions[0].level, Level.SECTION)


if __name__ == "__main__":
    unittest.main()
