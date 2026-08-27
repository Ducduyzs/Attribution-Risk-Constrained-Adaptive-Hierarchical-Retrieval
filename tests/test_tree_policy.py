from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from train_tree_policy import metrics, paper_sample_weights


class PaperWeightingTests(unittest.TestCase):
    def test_each_paper_has_equal_total_weight(self):
        rows = [
            {"source": "many"}, {"source": "many"}, {"source": "many"},
            {"source": "one"},
        ]
        weights = paper_sample_weights(rows)
        self.assertAlmostEqual(sum(weights[:3]), weights[3])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)

    def test_metrics_use_paper_weights(self):
        result = metrics(
            [0.9, 0.9, 0.9, 0.1], [1, 0, 0, 0], 0.5,
            paper_sample_weights([
                {"source": "many"}, {"source": "many"},
                {"source": "many"}, {"source": "one"},
            ]),
        )
        self.assertAlmostEqual(result["accuracy"], 0.6667, places=4)


if __name__ == "__main__":
    unittest.main()