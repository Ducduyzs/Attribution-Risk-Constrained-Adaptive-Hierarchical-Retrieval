from __future__ import annotations

import json
import sys
import tempfile
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
from edahr.models import _json_payload  # noqa: E402
from edahr.rollouts import RewardWeights  # noqa: E402
from edahr.schemas import DocumentSection, ScientificDocument  # noqa: E402
from edahr.training import sha256_file, v5_label, write_checkpoint_metadata  # noqa: E402


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
        self.assertGreater(weights.answer_quality, 0.0)
        self.assertGreater(weights.evidence_recall, 0.0)
        self.assertGreater(weights.citation_quality, 0.0)
        self.assertGreater(weights.attribution_risk_lambda, 0.0)
        self.assertGreater(weights.empty_evidence_lambda, 0.0)

    def test_v5_label_accepts_rescue_without_harm(self):
        row = {
            "branches": {
                "keep": {"reward": 0.1, "v5": {
                    "citation_precision": 1.0, "citation_recall": 0.5,
                    "harmful_rate": 0.0, "empty_evidence": 0,
                }},
                "parent": {"reward": 0.4, "v5": {
                    "citation_precision": 1.0, "citation_recall": 1.0,
                    "harmful_rate": 0.0, "empty_evidence": 0,
                }},
            }
        }
        self.assertEqual(v5_label(row, "parent", 0.02, 0.05, 0.02), 1)

    def test_v5_label_rejects_empty_or_recall_losing_expansion(self):
        keep = {"reward": 0.1, "v5": {
            "citation_precision": 1.0, "citation_recall": 1.0,
            "harmful_rate": 0.0, "empty_evidence": 0,
        }}
        alternatives = (
            {"reward": 0.5, "v5": {
                "citation_precision": 0.0, "citation_recall": 0.0,
                "harmful_rate": 0.0, "empty_evidence": 1,
            }},
            {"reward": 0.5, "v5": {
                "citation_precision": 1.0, "citation_recall": 0.5,
                "harmful_rate": 0.0, "empty_evidence": 0,
            }},
        )
        for expand in alternatives:
            row = {"branches": {"keep": keep, "parent": expand}}
            self.assertEqual(v5_label(row, "parent", 0.02, 0.05, 0.02), 0)


class CheckpointMetadataTests(unittest.TestCase):
    def test_metadata_records_hashes_and_v5_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "policy.ts"
            rollouts = root / "rollouts.jsonl"
            checkpoint.write_bytes(b"checkpoint")
            rollouts.write_text('{"query":"q"}\n', encoding="utf-8")
            output = write_checkpoint_metadata(
                checkpoint, {"val_auc": 0.75}, source_rollouts=rollouts,
                seed=7, min_margin=0.03, v5=True,
                epsilon=0.02, delta=0.05, tau=0.02,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(payload["source_rollouts_sha256"], sha256_file(rollouts))
            self.assertEqual(payload["feature_dim"], 14)
            self.assertTrue(payload["v5_constraints"]["enabled"])
            self.assertEqual(payload["training_report"]["val_auc"], 0.75)


class PreviewAgentParsingTests(unittest.TestCase):
    def test_antigravity_json_parser_accepts_fences_and_prose(self):
        fenced = '```json\n{"answerable": false, "claims": []}\n```'
        prose = 'Result follows: {"answerable": true, "claims": []} done.'
        self.assertFalse(_json_payload(fenced)["answerable"])
        self.assertTrue(_json_payload(prose)["answerable"])


if __name__ == "__main__":
    unittest.main()
