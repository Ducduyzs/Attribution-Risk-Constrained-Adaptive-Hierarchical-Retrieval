from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.agreement import agreement_ranking


class AgreementRankingTests(unittest.TestCase):
    def test_disagreeing_top_leaf_is_demoted(self):
        rerank = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        similarity = {"a": 0.1, "b": 0.9, "c": 0.8}  # "a" only the cross-encoder likes
        out = agreement_ranking(rerank, similarity)
        # RRF: b = 1/62+1/61 > a = 1/61+1/63 > c = 1/63+1/62
        self.assertEqual([leaf for leaf, _ in out], ["b", "a", "c"])

    def test_score_multiset_is_preserved(self):
        rerank = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        out = agreement_ranking(rerank, {"a": 0.2, "b": 0.3, "c": 0.9})
        self.assertEqual([score for _, score in out], [0.9, 0.5, 0.1])

    def test_agreement_keeps_order(self):
        rerank = [("a", 0.9), ("b", 0.5)]
        out = agreement_ranking(rerank, {"a": 0.9, "b": 0.1})
        self.assertEqual(out, rerank)

    def test_missing_similarity_fails(self):
        with self.assertRaises(KeyError):
            agreement_ranking([("a", 0.9)], {})


if __name__ == "__main__":
    unittest.main()
