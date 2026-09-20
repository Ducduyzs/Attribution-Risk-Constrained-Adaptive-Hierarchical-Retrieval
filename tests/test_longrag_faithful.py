from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.baselines_longrag import (
    LongRagFaithfulConfig,
    LongRagFaithfulIndex,
    LongRagFaithfulRetriever,
    answer_to_claims,
    build_document_units,
    build_reader_prompt,
)
from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.schemas import DocumentSection, ScientificDocument


def _fake_embed_factory(counter):
    import hashlib
    import random

    def embed(texts):
        counter[0] += 1
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1, 1) for _ in range(16)])
        return vectors

    return embed


def _hierarchy():
    settings = Settings(
        child_target_tokens=12, child_overlap_sentences=0,
        children_per_parent=4, parent_overlap_children=0,
    )
    para_a = "Alpha finding about retrieval precision here now."
    para_b = "Beta observation on reranking quality follows here."
    text_a = f"{para_a} {para_b}"
    documents = [
        ScientificDocument(
            document_id="doc0", source="a.pdf",
            sections=(
                DocumentSection(
                    "Results", text_a,
                    metadata={"paragraphs": [
                        {"paragraph_id": "a:p0", "text": para_a,
                         "char_start": 0, "char_end": len(para_a)},
                        {"paragraph_id": "a:p1", "text": para_b,
                         "char_start": len(para_a) + 1, "char_end": len(text_a)},
                    ]},
                ),
            ),
        ),
        ScientificDocument(
            document_id="doc1", source="b.pdf",
            sections=(
                DocumentSection(
                    "Results",
                    "Zeta unrelated material about tables. Eta filler closes.",
                ),
            ),
        ),
    ]
    return HierarchyBuilder(settings).build(documents)


def _config(**overrides):
    params = dict(
        embedding_model="fake-test", subchunk_tokens=6, top_k=1,
        reader_provider="shared-controlled",
    )
    params.update(overrides)
    return LongRagFaithfulConfig(**params)


class LongRagFaithfulTests(unittest.TestCase):
    def test_units_are_document_level_and_deterministic(self):
        hierarchy = _hierarchy()
        first = build_document_units(hierarchy)
        second = build_document_units(hierarchy)
        self.assertEqual([u.unit_id for u in first], ["a.pdf", "b.pdf"])
        self.assertEqual(
            [(u.unit_id, u.text) for u in first],
            [(u.unit_id, u.text) for u in second],
        )

    def test_units_preserve_paragraph_provenance(self):
        hierarchy = _hierarchy()
        units = {unit.unit_id: unit for unit in build_document_units(hierarchy)}
        paragraph_ids = {
            paragraph_id for _, _, paragraph_id, _ in units["a.pdf"].paragraphs
        }
        self.assertIn("a:p0", paragraph_ids)
        self.assertIn("a:p1", paragraph_ids)
        for start, end, _, text in units["a.pdf"].paragraphs:
            self.assertEqual(units["a.pdf"].text[start:end][: len(text)], text[: len(text)])

    def test_semantic_retriever_called_and_no_bm25(self):
        import edahr.baselines_longrag as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("Bm25", source)
        self.assertNotIn("mean(top", source)
        hierarchy = _hierarchy()
        counter = [0]
        index = LongRagFaithfulIndex(
            build_document_units(hierarchy), _config(),
            embed_fn=_fake_embed_factory(counter),
        )
        index.build()
        self.assertGreater(counter[0], 0)
        retriever = LongRagFaithfulRetriever(
            hierarchy, index, _config(), embed_fn=index._embed_fn
        )
        hits = retriever.search("retrieval precision", k=4)
        self.assertTrue(hits)
        for hit in hits:
            self.assertIn(hit.node_id, hierarchy.child_ids)

    def test_source_filtering(self):
        hierarchy = _hierarchy()
        index = LongRagFaithfulIndex(
            build_document_units(hierarchy), _config(),
            embed_fn=_fake_embed_factory([0]),
        )
        index.build()
        retriever = LongRagFaithfulRetriever(hierarchy, index, _config())
        hits = retriever.search("retrieval", k=4, source="b.pdf")
        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(hierarchy.node(hit.node_id).source, "b.pdf")

    def test_token_budget_topk_and_mapping_to_paragraphs(self):
        hierarchy = _hierarchy()
        config = _config(top_k=1)
        index = LongRagFaithfulIndex(
            build_document_units(hierarchy), config,
            embed_fn=_fake_embed_factory([0]),
        )
        index.build()
        retriever = LongRagFaithfulRetriever(hierarchy, index, config)
        trace = retriever.retrieval_trace("retrieval precision")
        # Token/top-k budget: exactly top_k units picked.
        self.assertEqual(len(trace["picked"]), 1)
        # Paragraph mapping resolves for whatever unit wins...
        paragraphs = retriever.paragraphs_for_units(
            [picked["unit_id"] for picked in trace["picked"]]
        )
        self.assertIsInstance(paragraphs, dict)
        # ...and the full unit set preserves provenance.
        all_paragraphs = retriever.paragraphs_for_units(["a.pdf", "b.pdf"])
        self.assertIn("a:p0", all_paragraphs)

    def test_index_save_load_identical_scores(self):
        hierarchy = _hierarchy()
        fake = _fake_embed_factory([0])
        index = LongRagFaithfulIndex(
            build_document_units(hierarchy), _config(), embed_fn=fake,
        )
        index.build()
        before = index.score_units("retrieval precision")
        with tempfile.TemporaryDirectory() as tmp:
            path = index.save(Path(tmp) / "index.pkl")
            reloaded = LongRagFaithfulIndex.load(path)
        # embed_fn is a transient callable and is not pickled; re-attach it.
        reloaded._embed_fn = fake
        after = reloaded.score_units("retrieval precision")
        self.assertEqual(before, after)

    def test_reader_prompt_and_claim_adapter(self):
        hierarchy = _hierarchy()
        units = build_document_units(hierarchy)
        prompt = build_reader_prompt("What is the finding?", units[:1])
        self.assertIn("[unit 1", prompt)
        self.assertIn("What is the finding?", prompt)
        claims = answer_to_claims(
            "Alpha finding about retrieval precision here now. Zeta tables.",
            [units[0].unit_id, units[1].unit_id],
            [units[0].text, units[1].text],
        )
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0][1], 0)

    def test_primary_mode_flag(self):
        self.assertEqual(LongRagFaithfulIndex.mode, "faithful")
        self.assertEqual(LongRagFaithfulRetriever.mode, "faithful")


if __name__ == "__main__":
    unittest.main()
