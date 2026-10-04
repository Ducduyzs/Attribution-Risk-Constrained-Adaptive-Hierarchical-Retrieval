from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.replay import INF, ReplayPolicy, replay_claim, replay_row
from edahr.risk_control import INF as RC_INF, check_monotone, crc_threshold, paper_folds


def candidate(child, support, retrieved, contradiction=0.0, coverage=0.0, lexical_block=False):
    return {
        "child_id": child, "nli_support": support, "nli_contradiction": contradiction,
        "lexical_coverage": coverage, "lexical_guard_blocked": lexical_block,
        "initially_retrieved": retrieved,
    }


def trace(*candidates, status="accepted"):
    return {"claim_text": "claim", "status": status, "candidates": list(candidates)}


class ReplayTests(unittest.TestCase):
    def test_legacy_guard_vetoes_entailment_fixed_does_not(self):
        claim = trace(candidate("a", 0.95, True, lexical_block=True))
        self.assertEqual(replay_claim(claim, ReplayPolicy(guard="legacy")), [])
        self.assertEqual(replay_claim(claim, ReplayPolicy(guard="fixed")), ["a"])

    def test_contradiction_always_vetoes(self):
        claim = trace(candidate("a", 0.95, True, contradiction=0.9, lexical_block=True))
        self.assertEqual(replay_claim(claim, ReplayPolicy(guard="fixed")), [])

    def test_expanded_threshold_controls_non_retrieved_leaves(self):
        claim = trace(candidate("new", 0.7, False), candidate("old", 0.4, True))
        self.assertEqual(replay_claim(claim, ReplayPolicy(expanded_threshold=0.5)), ["new"])
        self.assertEqual(replay_claim(claim, ReplayPolicy(expanded_threshold=0.8)), ["old"])
        self.assertEqual(replay_claim(claim, ReplayPolicy(expanded_threshold=INF)), ["old"])

    def test_row_decomposition(self):
        row = {
            "retrieved_child_ids": ["old"], "gold_child_ids": ["old"],
            "verification_trace": [trace(candidate("new", 0.9, False))],
        }
        out = replay_row(row, ReplayPolicy(expanded_threshold=0.5))
        self.assertEqual(out["harmful_drift"], ["new"])
        self.assertEqual(replay_row(row, ReplayPolicy(expanded_threshold=INF))["claims"], [])


class ConformalTests(unittest.TestCase):
    def test_threshold_satisfies_crc_inequality(self):
        # Unit i has loss 1 until lam reaches i/10, then 0.
        losses = [(lambda c: (lambda lam: 1.0 if lam < c else 0.0))(i / 10) for i in range(10)]
        grid = [i / 10 for i in range(11)]
        lam = crc_threshold(losses, grid, alpha=0.3)
        n = len(losses)
        risk = sum(loss(lam) for loss in losses) / n
        self.assertLessEqual(n / (n + 1) * risk + 1 / (n + 1), 0.3)
        smaller = [g for g in grid if g < lam]
        if smaller:
            prev = max(smaller)
            prev_risk = sum(loss(prev) for loss in losses) / n
            self.assertGreater(n / (n + 1) * prev_risk + 1 / (n + 1), 0.3)
        self.assertTrue(all(check_monotone(loss, grid) for loss in losses))

    def test_infeasible_alpha_falls_back_to_cite_retrieved_only(self):
        losses = [lambda lam: 1.0] * 3
        self.assertEqual(crc_threshold(losses, [0.5], alpha=0.1), RC_INF)

    def test_paper_folds_partition(self):
        folds = paper_folds(["p3", "p1", "p2", "p1"], 2)
        self.assertEqual(set().union(*folds), {"p1", "p2", "p3"})
        self.assertEqual(sum(len(f) for f in folds), 3)


if __name__ == "__main__":
    unittest.main()
