from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.statistics import (
    cluster_bootstrap_ci,
    cluster_sign_flip_test,
    compare,
    holm,
    non_inferiority,
    paired_effect_size,
)


class StatisticsTests(unittest.TestCase):
    def test_holm_matches_textbook_example(self):
        adjusted = holm([0.01, 0.04, 0.03, 0.005])
        self.assertEqual([round(p, 4) for p in adjusted], [0.03, 0.06, 0.06, 0.02])

    def test_sign_flip_exact_for_consistent_effect(self):
        # 8 papers all improving: only 2 of 2^8 sign patterns are as extreme.
        diffs = [0.1, 0.2] * 8
        clusters = [f"p{i // 2}" for i in range(16)]
        self.assertAlmostEqual(cluster_sign_flip_test(diffs, clusters), 2 / 256)
        self.assertAlmostEqual(
            cluster_sign_flip_test(diffs, clusters, alternative="greater"), 1 / 256
        )

    def test_null_effect_is_not_significant(self):
        diffs = [0.1, -0.1] * 10
        clusters = [f"p{i}" for i in range(20)]
        self.assertGreater(cluster_sign_flip_test(diffs, clusters), 0.5)

    def test_clustering_matters(self):
        # Same question-level diffs, but all positives in one paper.
        diffs = [0.3] * 5 + [0.0] * 5
        clustered = ["a"] * 5 + [f"b{i}" for i in range(5)]
        separate = [f"q{i}" for i in range(10)]
        self.assertGreater(
            cluster_sign_flip_test(diffs, clustered), cluster_sign_flip_test(diffs, separate)
        )

    def test_ci_and_effect_size(self):
        diffs = [0.05 + 0.01 * (i % 3) for i in range(30)]
        clusters = [f"p{i // 3}" for i in range(30)]
        low, high = cluster_bootstrap_ci(diffs, clusters)
        self.assertLess(low, 0.06)
        self.assertGreater(high, 0.06 - 1e-9)
        self.assertGreater(low, 0.0)
        self.assertEqual(paired_effect_size([0.1, 0.1], ["a", "b"]), 0.0)

    def test_non_inferiority(self):
        clusters = [f"p{i}" for i in range(12)]
        slightly_worse = [-0.005 + 0.002 * ((i % 3) - 1) for i in range(12)]
        self.assertTrue(non_inferiority(slightly_worse, clusters, margin=0.02)["non_inferior"])
        much_worse = [-0.05 + 0.002 * ((i % 3) - 1) for i in range(12)]
        self.assertFalse(non_inferiority(much_worse, clusters, margin=0.02)["non_inferior"])

    def test_compare_skips_undefined(self):
        out = compare({"q1": 0.5, "q2": None, "q3": 0.4}, {"q1": 0.3, "q2": 0.1, "q3": 0.4},
                      {"q1": "a", "q2": "a", "q3": "b"}, margin=0.02)
        self.assertEqual(out["questions"], 2)
        self.assertAlmostEqual(out["mean_diff"], 0.1)
        self.assertIn("non_inferiority", out)


if __name__ == "__main__":
    unittest.main()
