from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.context import assemble_context
from edahr.hierarchy import HierarchyBuilder
from edahr.schemas import DocumentSection, QueryType, ScientificDocument


def _hierarchy(two_sources=True):
    settings = Settings(
        child_target_tokens=12,
        child_overlap_sentences=0,
        children_per_parent=4,
        parent_overlap_children=0,
        final_context_k=6,
        context_token_budget=120,
        knapsack_token_bucket=16,
        context_dedup_threshold=0.8,
        max_source_share=0.5,
    )
    sections = [
        DocumentSection(
            "Results",
            "Alpha unique finding number one. Beta unique finding number two. "
            "Gamma unique finding number three. Delta unique finding four. "
            "Epsilon unique five here. Zeta unique six now.",
        )
    ]
    if two_sources:
        sections.append(
            DocumentSection(
                "Results",
                "Kappa other paper result one. Lambda other paper result two. "
                "Mu other paper result three. Nu other paper result four. "
                "Xi other paper five. Omicron other six.",
            )
        )
    documents = [
        ScientificDocument("doc-a", "a.pdf", tuple(sections[:1])),
        ScientificDocument("doc-b", "b.pdf", (sections[1],)) if two_sources else None,
    ]
    documents = [d for d in documents if d]
    return HierarchyBuilder(settings).build(documents), settings


class ContextGuardrailTests(unittest.TestCase):
    def test_strict_token_budget_even_for_single_oversized_block(self):
        hierarchy, settings = _hierarchy()
        # Only one candidate, budgeted strictly below its size.
        big_node = hierarchy.node(hierarchy.child_ids[0])
        tight_budget = max(1, big_node.token_count - 3)
        blocks = assemble_context(
            hierarchy, {big_node.node_id: 0.9}, QueryType.FACTOID,
            Settings(**{**settings.__dict__, "context_token_budget": tight_budget}),
        )
        self.assertEqual(len(blocks), 1)
        self.assertLessEqual(blocks[0].token_count, tight_budget)
        self.assertTrue(blocks[0].truncated)

    def test_knapsack_never_exceeds_budget(self):
        hierarchy, settings = _hierarchy()
        scores = {cid: 0.9 - i * 0.01 for i, cid in enumerate(hierarchy.child_ids)}
        blocks = assemble_context(hierarchy, scores, QueryType.FACTOID, settings)
        total = sum(block.token_count for block in blocks)
        self.assertLessEqual(total, settings.context_token_budget)
        # Utility-ordered prefix: the top candidate must be included.
        self.assertEqual(blocks[0].node_id, max(scores, key=scores.get))

    def test_near_duplicates_removed(self):
        hierarchy, settings = _hierarchy()
        children = list(hierarchy.child_ids)
        first = hierarchy.node(children[0])
        duplicate_id = "__dup__"
        scores = {
            duplicate_id: 0.95,          # same text as first child, higher score
            children[0]: 0.90,
            children[2]: 0.50,
        }
        patched = Settings(**{**settings.__dict__, "context_dedup_threshold": 0.99})
        hierarchy.nodes[duplicate_id] = type(first)(
            **{**first.__dict__, "node_id": duplicate_id}
        )
        blocks = assemble_context(hierarchy, scores, QueryType.FACTOID, patched)
        selected_nodes = {block.node_id for block in blocks}
        self.assertIn(duplicate_id, selected_nodes)
        self.assertNotIn(children[0], selected_nodes)

    def test_source_diversity_cap(self):
        hierarchy, settings = _hierarchy()
        scores = {cid: 0.9 - i * 0.01 for i, cid in enumerate(hierarchy.child_ids)}
        blocks = assemble_context(hierarchy, scores, QueryType.FACTOID, settings)
        per_source: dict[str, int] = {}
        for block in blocks:
            per_source[block.source] = per_source.get(block.source, 0) + 1
        for source, count in per_source.items():
            self.assertLessEqual(count / len(blocks), settings.max_source_share + 1e-6)

    def test_comparative_requires_multiple_sources(self):
        hierarchy, settings = _hierarchy()
        scores_a = {cid: 0.9 - i * 0.01 for i, cid in enumerate(hierarchy.child_ids)}
        # Give every doc-b child a slightly higher score so a single-source
        # greedy would pick only b.pdf; comparative must still surface a.pdf.
        node_b = [cid for cid in hierarchy.child_ids
                  if hierarchy.node(cid).source == "b.pdf"]
        scores = {cid: (0.95 if cid in node_b else 0.60)
                  for cid in hierarchy.child_ids}
        blocks = assemble_context(hierarchy, scores, QueryType.COMPARATIVE, settings)
        sources = {block.source for block in blocks}
        self.assertEqual(len(sources), 2)


if __name__ == "__main__":
    unittest.main()
