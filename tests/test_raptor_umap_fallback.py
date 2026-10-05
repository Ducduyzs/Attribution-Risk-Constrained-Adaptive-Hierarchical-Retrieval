from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr import baselines_raptor as B


class UmapFallbackTests(unittest.TestCase):
    def test_local_umap_failure_keeps_cluster_whole(self):
        try:
            import umap
        except ImportError:
            self.skipTest("umap-learn not installed")
        real = umap.UMAP
        calls = {"n": 0}

        class FailingLocal:
            """First (global) UMAP works; every local UMAP raises like an empty graph."""

            def __init__(self, *args, **kwargs):
                calls["n"] += 1
                self.inner = real(*args, **kwargs) if calls["n"] == 1 else None

            def fit_transform(self, matrix):
                if self.inner is None:
                    raise ValueError("zero-size array to reduction operation maximum")
                return self.inner.fit_transform(matrix)

        import numpy as np
        rng = np.random.default_rng(0)
        # Two tight groups of 20 so global clusters exceed dim + 1 and the
        # local UMAP stage actually runs (and fails).
        centers = [np.eye(16)[0] * 10, np.eye(16)[1] * 10]
        vectors = [list(centers[i // 20] + rng.normal(scale=0.05, size=16)) for i in range(40)]
        before = B.UMAP_FALLBACKS[0]
        # Force one global cluster of all 40 points so the local stage must run.
        one_cluster = lambda vectors, *a, **k: ([[0]] * len(vectors), 1)
        with mock.patch.object(umap, "UMAP", FailingLocal),                 mock.patch.object(B, "_gmm_bic_labels", one_cluster):
            membership = B.raptor_cluster_indices(vectors)
        self.assertEqual(set(map(tuple, membership)), {(0,)})  # whole cluster kept
        self.assertEqual(len(membership), 40)
        self.assertTrue(all(membership))
        self.assertGreater(calls["n"], 1, "local UMAP stage was not exercised")
        self.assertGreater(B.UMAP_FALLBACKS[0], before)


if __name__ == "__main__":
    unittest.main()
