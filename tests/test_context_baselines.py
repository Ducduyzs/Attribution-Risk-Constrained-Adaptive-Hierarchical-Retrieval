from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.baselines import Bm25ChildRetriever, auto_label_gold_children, run_benchmark
from edahr.config import Settings
from edahr.context_baselines import FullDocumentPipeline, OracleEvidencePipeline
from edahr.hierarchy import HierarchyBuilder
from edahr.pipeline import AdaptiveHierarchicalPipeline
from edahr.schemas import Claim, DocumentSection, Generation, Level, ScientificDocument


class FakeReranker:
    def score(self, query, texts):
        return [0.9 for _ in texts]


class RecordingGenerator:
    """Cites the first context block and remembers every context it saw."""

    def __init__(self):
        self.contexts = []

    def generate(self, query, context):
        self.contexts.append(list(context))
        return Generation(True, (Claim("gold finding confirmed", (context[0].context_id,), 0.9),))


class FakeVerifier:
    def support_score(self, claim, evidence):
        return 0.95 if "gold" in evidence.lower() else 0.1


SECTIONS = (
    DocumentSection(
        "Methods",
        "Filler beta text without marker. Gold finding alpha here now. "
        "More filler delta content. Another epsilon sentence.",
    ),
    DocumentSection(
        "Results",
        "Unrelated zeta preamble here. Another eta sentence follows. "
        "Theta unrelated material continues. Iota filler closes section.",
    ),
)
RECORD = {
    "query": "What is the gold finding?",
    "source": "doc.pdf",
    "gold_quotes": ["Gold finding alpha here now."],
    "answer": "gold finding confirmed",
}


def _setup():
    settings = Settings(
        child_target_tokens=10, child_overlap_sentences=0, children_per_parent=2,
        parent_overlap_children=0, min_child_hits=1, final_context_k=2,
        context_token_budget=500, merge_margin=-1.0,
    )
    hierarchy = HierarchyBuilder(settings).build([
        ScientificDocument(document_id="doc", source="doc.pdf", sections=SECTIONS),
        ScientificDocument(document_id="other", source="other.pdf", sections=SECTIONS[1:]),
    ])
    return hierarchy, settings


def _components(hierarchy):
    return dict(
        retriever=Bm25ChildRetriever(hierarchy), reranker=FakeReranker(),
        generator=RecordingGenerator(), verifier=FakeVerifier(),
    )


class OracleEvidenceTests(unittest.TestCase):
    def test_context_is_exactly_the_gold_leaves(self):
        hierarchy, settings = _setup()
        gold, _ = auto_label_gold_children(hierarchy, RECORD)
        parts = _components(hierarchy)
        pipeline = OracleEvidencePipeline(
            hierarchy=hierarchy, settings=settings,
            gold_children={("doc.pdf", RECORD["query"]): sorted(gold)}, **parts,
        )
        run = run_benchmark("oracle", pipeline, [RECORD], ks=(1,))
        context = parts["generator"].contexts[0]
        self.assertEqual({block.node_id for block in context}, gold)
        self.assertTrue(all(block.level == Level.CHILD for block in context))
        row = run.rows[0]
        self.assertTrue(row["retrieval_free"])
        self.assertIsNone(row["recall@1"])
        self.assertIsNone(row["mrr"])
        self.assertNotIn("recall@1", run.summary)
        self.assertEqual(row["harmful_drift_leaf_ids"], [])
        self.assertEqual(row["citation_f1"], 1.0)

    def test_unknown_question_fails_loudly(self):
        hierarchy, settings = _setup()
        pipeline = OracleEvidencePipeline(
            hierarchy=hierarchy, settings=settings, gold_children={}, **_components(hierarchy),
        )
        with self.assertRaises(KeyError):
            pipeline.answer("unregistered", source="doc.pdf")


class FullDocumentTests(unittest.TestCase):
    def test_context_is_every_section_of_the_paper_in_order(self):
        hierarchy, settings = _setup()
        parts = _components(hierarchy)
        pipeline = FullDocumentPipeline(
            hierarchy=hierarchy, settings=settings, token_budget=10_000, **parts,
        )
        result = pipeline.answer(RECORD["query"], source="doc.pdf")
        context = parts["generator"].contexts[0]
        # final_context_k=2 in settings must not cap the whole-paper context.
        self.assertEqual([block.level for block in context], [Level.SECTION, Level.SECTION])
        self.assertEqual([block.text[:7] for block in context], ["Filler ", "Unrelat"])
        self.assertEqual({block.source for block in context}, {"doc.pdf"})
        self.assertFalse(any(block.truncated for block in context))
        self.assertEqual(result.metrics["context_dropped_nodes"], 0.0)
        # Citations of a section still resolve to leaf children.
        self.assertTrue(all(
            hierarchy.node(evidence.node_id).level == Level.CHILD
            for evidence in result.evidence.values()
        ))

    def test_requires_source(self):
        hierarchy, settings = _setup()
        pipeline = FullDocumentPipeline(
            hierarchy=hierarchy, settings=settings, **_components(hierarchy),
        )
        with self.assertRaises(ValueError):
            pipeline.answer(RECORD["query"], source=None)


class ProvenanceAccuracyTests(unittest.TestCase):
    def test_undefined_without_gold_pages(self):
        hierarchy, settings = _setup()
        pipeline = AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, settings=settings, **_components(hierarchy),
        )
        run = run_benchmark("no-pages", pipeline, [RECORD], ks=(1,))
        self.assertIsNone(run.rows[0]["provenance_accuracy"])
        self.assertNotIn("provenance_accuracy", run.summary)
        self.assertFalse(run.rows[0]["retrieval_free"])
        self.assertIsNotNone(run.rows[0]["recall@1"])


if __name__ == "__main__":
    unittest.main()
