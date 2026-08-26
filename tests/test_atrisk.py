from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.attribution import (  # noqa: E402
    attribution_metrics,
    attribution_risk,
    citation_survival_rate,
    unsupported_claim_rate,
)
from edahr.config import Settings  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.invariants import validate_hierarchy  # noqa: E402
from edahr.rollouts import RewardWeights  # noqa: E402
from edahr.schemas import DocumentSection, ScientificDocument  # noqa: E402


class AttributionRiskTests(unittest.TestCase):
    def test_ar_perfect_support_is_zero(self):
        supports = [("c1", 1.0), ("c2", 0.9)]
        self.assertAlmostEqual(attribution_risk(supports), 0.05)

    def test_ar_no_supports_is_max_risk(self):
        self.assertEqual(attribution_risk([]), 1.0)

    def test_unsupported_rate_and_survival(self):
        supports = [("c1", 0.9), ("c2", 0.1)]
        self.assertAlmostEqual(unsupported_claim_rate(supports, 0.25), 0.5)
        self.assertAlmostEqual(citation_survival_rate(4, 1), 0.25)
        metrics = attribution_metrics(supports, generated_claims=2, verified_claims=1, nli_threshold=0.25)
        self.assertAlmostEqual(metrics["attribution_risk"], 0.5)
        self.assertAlmostEqual(metrics["citation_survival_rate"], 0.5)


class HierarchyInvariantTests(unittest.TestCase):
    def test_synthetic_hierarchy_has_no_violations(self):
        settings = Settings(
            child_target_tokens=12,
            child_overlap_sentences=0,
            children_per_parent=2,
            parent_overlap_children=0,
            min_child_hits=2,
        )
        document = ScientificDocument(
            document_id="doc",
            source="doc.pdf",
            sections=(
                DocumentSection("Results", "One two three four. Five six seven eight. Nine ten eleven twelve.", page_start=1, page_end=2),
                DocumentSection("Method", "Alpha beta gamma delta. Epsilon zeta eta theta.", page_start=2, page_end=3),
            ),
        )
        hierarchy = HierarchyBuilder(settings).build([document])
        self.assertEqual(validate_hierarchy(hierarchy), [])

    def test_tampered_child_pages_are_detected(self):
        settings = Settings(
            child_target_tokens=12,
            child_overlap_sentences=0,
            children_per_parent=2,
            parent_overlap_children=0,
            min_child_hits=2,
        )
        document = ScientificDocument(
            document_id="doc",
            source="doc.pdf",
            sections=(DocumentSection("Results", "One two three four. Five six seven eight."),),
        )
        hierarchy = HierarchyBuilder(settings).build([document])
        child_id = hierarchy.child_ids[0]
        tampered = hierarchy.nodes[child_id].replaced(page_start=9, page_end=11)
        hierarchy.nodes[child_id] = tampered
        issues = validate_hierarchy(hierarchy)
        self.assertTrue(any("pages" in issue for issue in issues))


class RolloutRowTests(unittest.TestCase):
    def test_reward_weights_defaults_match_spec(self):
        weights = RewardWeights()
        self.assertEqual(weights.answer_quality, 0.0)      # enabled with QASPER gold
        self.assertGreater(weights.evidence_recall, 0.0)
        self.assertGreater(weights.citation_quality, 0.0)
        self.assertGreater(weights.attribution_risk_lambda, 0.0)


if __name__ == "__main__":
    unittest.main()
