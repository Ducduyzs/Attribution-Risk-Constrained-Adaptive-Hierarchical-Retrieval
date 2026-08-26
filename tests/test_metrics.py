from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.evaluation import citation_precision, evidence_density, recall_at_k, reciprocal_rank


class MetricTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        retrieved = ["a", "b", "c"]
        self.assertEqual(recall_at_k(retrieved, {"b", "x"}, 2), 0.5)
        self.assertEqual(reciprocal_rank(retrieved, {"b"}), 0.5)

    def test_grounding_metrics(self):
        self.assertEqual(citation_precision(["E1", "E2"], {"E2"}), 0.5)
        self.assertEqual(evidence_density(40, 100), 0.4)


if __name__ == "__main__":
    unittest.main()
