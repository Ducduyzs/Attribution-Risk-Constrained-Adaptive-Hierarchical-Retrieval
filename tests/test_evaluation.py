from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.evaluation import (
    aggregate,
    answer_exact_match,
    answer_token_f1,
    aurc,
    bootstrap_ci,
    citation_f1,
    citation_precision,
    citation_recall,
    e_aurc,
    evidence_span_recall,
    hit_rate_at_k,
    latency_stats,
    ndcg_at_k,
    paired_bootstrap_test,
    precision_at_k,
    provenance_accuracy,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)


class RankingMetricsTests(unittest.TestCase):
    def test_ndcg_perfect_and_imperfect(self):
        perfect = ndcg_at_k(["a", "b", "c"], {"a": 3.0, "b": 2.0}, 3)
        self.assertAlmostEqual(perfect, 1.0)
        swapped = ndcg_at_k(["b", "a"], {"a": 3.0, "b": 1.0}, 2)
        self.assertLess(swapped, 1.0)
        self.assertGreater(swapped, 0.5)
        self.assertEqual(ndcg_at_k(["x"], {"a": 1.0}, 1), 0.0)

    def test_precision_hit_rate(self):
        self.assertEqual(precision_at_k(["a", "z", "b"], {"a", "b"}, 3), 2 / 3)
        self.assertEqual(hit_rate_at_k(["z"], {"a"}, 1), 0.0)
        self.assertEqual(hit_rate_at_k(["z", "a"], {"a"}, 2), 1.0)

    def test_span_recall_with_segmentation_noise(self):
        gold = ["dense retrieval finds semantic matches"]
        predicted = ["dense retrieval finds semantic matches . extra context"]
        self.assertEqual(evidence_span_recall(predicted, gold, tau=0.6), 1.0)
        self.assertEqual(evidence_span_recall([], gold), 0.0)
        self.assertEqual(evidence_span_recall(predicted, []), 0.0)


class GroundingMetricTests(unittest.TestCase):
    def test_citation_precision_recall_f1(self):
        predicted = ["E1", "E2", "E9"]
        supported = {"E1", "E3"}
        self.assertAlmostEqual(citation_precision(predicted, supported), 1 / 3)
        self.assertAlmostEqual(citation_recall(predicted, supported), 0.5)
        expected_f1 = 2 * (1 / 3) * 0.5 / (1 / 3 + 0.5)
        self.assertAlmostEqual(citation_f1(predicted, supported), expected_f1)

    def test_provenance_page_range(self):
        predictions = [("a.pdf", 3, 4), ("a.pdf", 8, 8), ("b.pdf", 6, 8)]
        gold = {("a.pdf", 4), ("b.pdf", 7)}
        self.assertAlmostEqual(provenance_accuracy(predictions, gold), 2 / 3)
        self.assertEqual(provenance_accuracy([], gold), 0.0)

    def test_answer_metrics(self):
        self.assertEqual(answer_exact_match("The CAT sat", "the cat sat"), 1.0)
        self.assertGreater(answer_token_f1("cat sat on mat", "cat on the mat"), 0.6)


class SelectivePredictionTests(unittest.TestCase):
    def test_risk_coverage_monotone_confidence(self):
        accuracies = [1.0, 0.0, 1.0]
        confidences = [0.9, 0.8, 0.7]
        coverages, risks = risk_coverage_curve(accuracies, confidences)
        self.assertEqual(coverages[0], 1 / 3)
        self.assertAlmostEqual(risks[0], 0.0)          # most confident correct
        self.assertAlmostEqual(risks[-1], 1 / 3)

    def test_eaurc_rewards_better_ordering(self):
        good_acc, good_conf = [1.0, 1.0, 0.0], [0.9, 0.8, 0.1]
        bad_acc, bad_conf = [0.0, 1.0, 1.0], [0.9, 0.8, 0.1]
        self.assertLess(e_aurc(good_acc, good_conf), e_aurc(bad_acc, bad_conf))
        self.assertLessEqual(aurc(good_acc, good_conf), aurc(bad_acc, bad_conf))

    def test_selective_accuracy_at_coverage(self):
        accuracies = [1.0, 1.0, 0.0, 0.0]
        confidences = [0.95, 0.85, 0.5, 0.4]
        value = selective_accuracy_at_coverage(accuracies, confidences, 0.5)
        self.assertEqual(value, 1.0)


class SystemStatisticsTests(unittest.TestCase):
    def test_latency_percentiles(self):
        stats = latency_stats([10.0, 20.0, 30.0, 40.0, 100.0])
        self.assertAlmostEqual(stats["mean_ms"], 40.0)
        self.assertEqual(stats["max_ms"], 100.0)
        self.assertGreaterEqual(stats["p95_ms"], 40.0)

    def test_bootstrap_ci_brackets_mean(self):
        samples = [0.8, 0.82, 0.79, 0.81, 0.83, 0.80]
        low, high = bootstrap_ci(samples, iterations=400, seed=1)
        mean = sum(samples) / len(samples)
        self.assertLessEqual(low, mean)
        self.assertLessEqual(mean, high)

    def test_paired_bootstrap_detects_difference(self):
        first = [0.9] * 12
        second = [0.4] * 12
        p_value = paired_bootstrap_test(first, second, iterations=300, seed=3)
        self.assertLess(p_value, 0.05)
        same_p = paired_bootstrap_test([0.5] * 8, [0.5] * 8, iterations=200, seed=5)
        self.assertGreater(same_p, 0.05)

    def test_aggregate_macro_average(self):
        rows = [{"recall@5": 1.0, "note": "text"}, {"recall@5": 0.0}]
        summary = aggregate(rows)
        self.assertAlmostEqual(summary["recall@5"], 0.5)
        self.assertNotIn("note", summary)


if __name__ == "__main__":
    unittest.main()
