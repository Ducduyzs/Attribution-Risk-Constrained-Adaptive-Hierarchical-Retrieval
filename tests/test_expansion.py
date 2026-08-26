from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.expansion import expand_selection, max_level_for_query
from edahr.hierarchy import HierarchyBuilder
from edahr.policy import AdaptiveMergePolicy
from edahr.schemas import (
    DocumentSection,
    Level,
    QueryType,
    ScientificDocument,
    level_rank,
)


class LevelAwareReranker:
    """High scores for long section/document texts, low for parents/children.

    Node.text carries no level prefix (parents are joins of children and
    sections are normalized bodies), so length is the reliable discriminator
    for this fixture: children ~30 chars, parents ~60, sections 150+,
    documents 300+.
    """

    def __init__(self, high=0.95, low=0.30, threshold=100):
        self.high = high
        self.low = low
        self.threshold = threshold

    def score(self, query, texts):
        return [
            self.high if len(text) > self.threshold else self.low
            for text in texts
        ]


def _hierarchy():
    settings = Settings(
        child_target_tokens=12,
        child_overlap_sentences=0,
        children_per_parent=2,
        parent_overlap_children=0,
        min_child_hits=2,
        context_token_budget=4000,
        expansion_max_depth=3,
    )
    document = ScientificDocument(
        document_id="doc",
        source="doc.pdf",
        sections=(
            DocumentSection(
                "Results",
                "Alpha result one here. Beta result two there. Gamma result three now. "
                "Delta result four then. Epsilon result five late. Zeta result six early.",
            ),
            DocumentSection(
                "Discussion",
                "Eta discussion point one. Theta discussion point two. "
                "Iota discussion point three. Kappa discussion point four.",
            ),
        ),
    )
    return HierarchyBuilder(settings).build([document]), settings


def _selected_children(hierarchy, score=0.35):
    return {cid: score for cid in hierarchy.child_ids}


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.hierarchy, self.settings = _hierarchy()

    def test_factoid_ceiling_is_parent(self):
        self.assertEqual(max_level_for_query(QueryType.FACTOID), Level.PARENT)
        policy = AdaptiveMergePolicy(margin=-1.0)  # always allow merges
        selected = _selected_children(self.hierarchy)
        # First merge children into parents (start_rank=0).
        merged, decisions, trace = expand_selection(
            "q", QueryType.FACTOID, self.hierarchy, LevelAwareReranker(),
            policy, self.settings, dict(selected), start_rank=0,
        )
        levels_used = {
            node.level for node in (self.hierarchy.nodes[n] for n in merged)
        }
        self.assertNotIn(Level.SECTION, levels_used)
        self.assertNotIn(Level.DOCUMENT, levels_used)

    def test_global_expands_to_document(self):
        policy = AdaptiveMergePolicy(margin=-1.0, cost_penalty=0.0)
        selected = _selected_children(self.hierarchy)
        final, decisions, trace = expand_selection(
            "q", QueryType.GLOBAL, self.hierarchy, LevelAwareReranker(),
            policy, self.settings, dict(selected), start_rank=0,
        )
        nodes = [self.hierarchy.nodes[node_id] for node_id in final]
        self.assertTrue(any(node.level is Level.DOCUMENT for node in nodes))
        joined_trace = "\n".join(trace)
        self.assertIn("section", joined_trace)
        self.assertIn("document", joined_trace)

    def test_early_stop_when_no_candidate_benefits(self):
        class PessimisticReranker(LevelAwareReranker):
            def score(self, query, texts):
                return [self.low for _ in texts]

        policy = AdaptiveMergePolicy(margin=5.0)
        _, decisions, trace = expand_selection(
            "q", QueryType.GLOBAL, self.hierarchy, PessimisticReranker(),
            policy, self.settings, _selected_children(self.hierarchy), start_rank=1,
        )
        self.assertFalse([d for d in decisions if d.accepted])
        self.assertTrue(trace)

    def test_budget_headroom_guard(self):
        tight_settings = Settings(
            child_target_tokens=12,
            child_overlap_sentences=0,
            children_per_parent=2,
            parent_overlap_children=0,
            min_child_hits=2,
            context_token_budget=10,
            expansion_headroom=1.0,
            expansion_max_depth=3,
        )
        policy = AdaptiveMergePolicy(margin=-1.0)
        _, _, trace = expand_selection(
            "q", QueryType.GLOBAL, self.hierarchy, LevelAwareReranker(),
            policy, tight_settings, _selected_children(self.hierarchy), start_rank=1,
        )
        self.assertTrue(any("headroom" in step for step in trace))


if __name__ == "__main__":
    unittest.main()
