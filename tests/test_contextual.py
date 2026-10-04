from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.config import Settings
from edahr.contextual import apply_chunk_context, context_key, document_text
from edahr.hierarchy import HierarchyBuilder
from edahr.schemas import DocumentSection, ScientificDocument


def _hierarchy():
    settings = Settings(child_target_tokens=10, child_overlap_sentences=0,
                        children_per_parent=2, parent_overlap_children=0)
    document = ScientificDocument(
        document_id="doc", source="doc.qasper", metadata={"title": "A Study of Gold"},
        sections=(DocumentSection("Results", "Gold finding alpha here now. Filler beta text."),),
    )
    return HierarchyBuilder(settings).build([document])


class ContextualTests(unittest.TestCase):
    def test_none_is_identity(self):
        hierarchy = _hierarchy()
        self.assertIs(apply_chunk_context(hierarchy, "none"), hierarchy)

    def test_title_changes_only_retrieval_text(self):
        hierarchy = _hierarchy()
        out = apply_chunk_context(hierarchy, "title")
        self.assertEqual(out.child_ids, hierarchy.child_ids)
        for child_id in hierarchy.child_ids:
            node, new = hierarchy.node(child_id), out.node(child_id)
            self.assertEqual(new.text, node.text)
            self.assertTrue(new.embedding_text.startswith("Paper: A Study of Gold\nSection: Results"))
            self.assertTrue(new.embedding_text.endswith(node.text))

    def test_llm_mode_uses_cache_and_rejects_stale_entries(self):
        hierarchy = _hierarchy()
        model = "m"
        document = document_text(hierarchy, "doc.qasper")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ctx.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for child_id in hierarchy.child_ids:
                    handle.write(json.dumps({
                        "child_id": child_id, "context": "Situating context.",
                        "key": context_key(model, document, hierarchy.node(child_id).text),
                    }) + "\n")
            out = apply_chunk_context(hierarchy, "llm", path, model)
            first = out.node(hierarchy.child_ids[0])
            self.assertTrue(first.embedding_text.startswith("Situating context.\n\n"))
            with self.assertRaises(RuntimeError):
                apply_chunk_context(hierarchy, "llm", path, "other-model")
            with self.assertRaises(RuntimeError):
                apply_chunk_context(hierarchy, "llm", Path(directory) / "missing.jsonl", model)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            apply_chunk_context(_hierarchy(), "bogus")


if __name__ == "__main__":
    unittest.main()
