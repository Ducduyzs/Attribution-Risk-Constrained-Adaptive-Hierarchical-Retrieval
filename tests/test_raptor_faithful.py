from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.baselines_raptor import (
    RaptorFaithfulConfig,
    RaptorFaithfulRetriever,
    build_paper_tree,
    load_tree,
    raptor_cluster_indices,
    save_tree,
)
from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.schemas import DocumentSection, ScientificDocument


def _fake_embed(texts):
    vectors = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = __import__("random").Random(seed)
        vectors.append([rng.uniform(-1, 1) for _ in range(16)])
    return vectors


def _fake_summarizer_factory(counter):
    def summarize(context):
        counter[0] += 1
        return "Summary of: " + " ".join(context.split()[:30])

    return summarize


def _hierarchy(sources=("a.pdf", "b.pdf"), children_texts=None):
    settings = Settings(
        child_target_tokens=12, child_overlap_sentences=0,
        children_per_parent=4, parent_overlap_children=0,
    )
    if children_texts is None:
        children_texts = [
            "Alpha finding about retrieval precision here now. ",
            "Beta observation on reranking quality follows here. ",
            "Gamma result concerning attribution drift appears. ",
            "Delta note about token budget limits discussed. ",
            "Epsilon claim regarding evidence density stated. ",
            "Zeta remark about hierarchical merging written. ",
        ]
    documents = []
    for index, source in enumerate(sources):
        text = " ".join(children_texts)
        documents.append(
            ScientificDocument(
                document_id=f"doc{index}", source=source,
                sections=(DocumentSection("Results", text),),
            )
        )
    return HierarchyBuilder(settings).build(documents)


def _config(**overrides):
    params = dict(
        embedding_model="fake-test", summarizer_model="fake-test",
        seed=224, cache_dir="", device="cpu",
    )
    params.update(overrides)
    return RaptorFaithfulConfig(**params)


class FaithfulTreeTests(unittest.TestCase):
    def test_tree_covers_all_leaves(self):
        hierarchy = _hierarchy(sources=("a.pdf",))
        tree, meta = build_paper_tree(
            hierarchy, "a.pdf", _config(),
            _fake_summarizer_factory([0]), embed_fn=_fake_embed,
        )
        covered = {c for node in tree.nodes for c in node.member_child_ids}
        self.assertEqual(covered, set(tree.leaf_child_ids))
        self.assertEqual(set(tree.leaf_child_ids),
                         {c for c in hierarchy.child_ids
                          if hierarchy.node(c).source == "a.pdf"})

    def test_recursion_creates_multiple_layers(self):
        hierarchy = _hierarchy(sources=("a.pdf",))
        tree, _ = build_paper_tree(
            hierarchy, "a.pdf", _config(max_layers=5),
            _fake_summarizer_factory([0]), embed_fn=_fake_embed,
        )
        self.assertGreaterEqual(tree.layers, 2)
        self.assertTrue(any(node.layer > 0 for node in tree.nodes))

    def test_summaries_are_cached(self):
        hierarchy = _hierarchy(sources=("a.pdf",))
        with tempfile.TemporaryDirectory() as cache:
            counter = [0]
            _, first = build_paper_tree(
                hierarchy, "a.pdf", _config(), _fake_summarizer_factory(counter),
                embed_fn=_fake_embed, cache_dir=cache,
            )
            self.assertEqual(first["cache"], "miss")
            calls_after_first = counter[0]
            self.assertGreater(calls_after_first, 0)
            _, second = build_paper_tree(
                hierarchy, "a.pdf", _config(), _fake_summarizer_factory(counter),
                embed_fn=_fake_embed, cache_dir=cache,
            )
            self.assertEqual(second["cache"], "hit")
            self.assertEqual(counter[0], calls_after_first)

    def test_clustering_is_deterministic(self):
        vectors = _fake_embed([f"document sentence number {i}" for i in range(12)])
        first = raptor_cluster_indices(vectors, seed=224)
        second = raptor_cluster_indices(vectors, seed=224)
        self.assertEqual(first, second)

    def test_save_load_identical_retrieval(self):
        hierarchy = _hierarchy(sources=("a.pdf",))
        tree, _ = build_paper_tree(
            hierarchy, "a.pdf", _config(),
            _fake_summarizer_factory([0]), embed_fn=_fake_embed,
        )
        retriever = RaptorFaithfulRetriever(
            hierarchy, {"a.pdf": tree}, _config(), embed_fn=_fake_embed
        )
        before = [h.node_id for h in retriever.search("retrieval precision", k=4)]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_tree(tree, Path(tmp) / "tree.pkl")
            reloaded = load_tree(str(path))
        retriever2 = RaptorFaithfulRetriever(
            hierarchy, {"a.pdf": reloaded}, _config(), embed_fn=_fake_embed
        )
        after = [h.node_id for h in retriever2.search("retrieval precision", k=4)]
        self.assertEqual(before, after)

    def test_retrieval_considers_internal_nodes_and_maps_to_leaves(self):
        hierarchy = _hierarchy(sources=("a.pdf",))
        tree, _ = build_paper_tree(
            hierarchy, "a.pdf", _config(),
            _fake_summarizer_factory([0]), embed_fn=_fake_embed,
        )
        retriever = RaptorFaithfulRetriever(
            hierarchy, {"a.pdf": tree}, _config(), embed_fn=_fake_embed
        )
        trace = retriever.retrieval_trace("attribution drift")
        self.assertGreater(trace["internal_nodes_scored"], 0)
        hits = retriever.search("attribution drift", k=4)
        self.assertTrue(hits)
        for hit in hits:
            self.assertIn(hit.node_id, hierarchy.child_ids)

    def test_source_filtering(self):
        hierarchy = _hierarchy(sources=("a.pdf", "b.pdf"))
        trees = {}
        for source in ("a.pdf", "b.pdf"):
            tree, _ = build_paper_tree(
                hierarchy, source, _config(),
                _fake_summarizer_factory([0]), embed_fn=_fake_embed,
            )
            trees[source] = tree
        retriever = RaptorFaithfulRetriever(
            hierarchy, trees, _config(), embed_fn=_fake_embed
        )
        hits = retriever.search("retrieval", k=6, source="b.pdf")
        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(hierarchy.node(hit.node_id).source, "b.pdf")

    def test_primary_mode_is_semantic_not_lexical(self):
        import edahr.baselines_raptor as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("TfidfVectorizer", source)
        self.assertNotIn("_LexicalScorer", source)
        self.assertEqual(RaptorFaithfulRetriever.mode, "faithful")


if __name__ == "__main__":
    unittest.main()
