from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_benchmark_failures import classify_failure


def row(**overrides):
    value = {
        "citation_evaluable": True,
        "gold_child_ids": ["gold"],
        "candidate_child_ids": ["gold"],
        "evidence_node_ids": [],
        "hit_rate@10": 1.0,
    }
    value.update(overrides)
    return value


class FailureClassificationTests(unittest.TestCase):
    def test_observable_pipeline_failures_are_mutually_exclusive(self):
        self.assertEqual(
            classify_failure(row(**{"hit_rate@10": 0.0})), "retrieval_miss_at_10"
        )
        self.assertEqual(
            classify_failure(row(candidate_child_ids=["wrong"])),
            "candidate_selection_miss",
        )
        self.assertEqual(classify_failure(row()), "post_context_no_evidence")
        self.assertEqual(
            classify_failure(row(generated_claim_count=0)), "generation_empty"
        )
        self.assertEqual(
            classify_failure(row(generated_claim_count=2)), "verifier_rejected_all"
        )
        self.assertEqual(
            classify_failure(row(evidence_node_ids=["wrong"])), "wrong_citation_only"
        )

    def test_success_distinguishes_partial_and_full_gold_coverage(self):
        self.assertEqual(
            classify_failure(row(evidence_node_ids=["gold"])),
            "full_grounding_success",
        )
        self.assertEqual(
            classify_failure(row(
                gold_child_ids=["gold", "other"], evidence_node_ids=["gold"]
            )),
            "partial_grounding_success",
        )


if __name__ == "__main__":
    unittest.main()