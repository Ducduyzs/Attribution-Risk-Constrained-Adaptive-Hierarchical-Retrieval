from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.scifact import convert_scifact


class SciFactConversionTests(unittest.TestCase):
    def test_evidence_pair_has_stable_identity_label_and_quote_union(self):
        corpus = [{
            "doc_id": 7, "title": "Paper", "abstract": ["First.", "Gold A.", "Gold B."],
            "structured": False,
        }]
        claims = [{
            "id": 3, "claim": "A finding.", "cited_doc_ids": [7],
            "evidence": {"7": [
                {"sentences": [2], "label": "CONTRADICT"},
                {"sentences": [1, 2], "label": "CONTRADICT"},
            ]},
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, claims_path = root / "corpus.jsonl", root / "claims.jsonl"
            corpus_path.write_text(json.dumps(corpus[0]) + "\n", encoding="utf-8")
            claims_path.write_text(json.dumps(claims[0]) + "\n", encoding="utf-8")
            papers, questions, report = convert_scifact(corpus_path, claims_path)

        self.assertEqual(report["evaluable_claim_document_pairs"], 1)
        self.assertEqual(papers[0]["source"], "7.scifact")
        self.assertEqual(questions[0]["question_id"], "scifact-3-7")
        self.assertEqual(questions[0]["answer"], "contradicted")
        self.assertEqual(questions[0]["gold_quotes"], ["Gold A.", "Gold B."])

    def test_claims_without_evidence_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, claims_path = root / "corpus.jsonl", root / "claims.jsonl"
            corpus_path.write_text(
                json.dumps({"doc_id": 7, "title": "P", "abstract": ["A."]}) + "\n",
                encoding="utf-8",
            )
            claims_path.write_text(
                json.dumps({"id": 1, "claim": "C", "evidence": {}, "cited_doc_ids": [7]}) + "\n",
                encoding="utf-8",
            )
            papers, questions, _ = convert_scifact(corpus_path, claims_path)
        self.assertEqual((papers, questions), ([], []))


if __name__ == "__main__":
    unittest.main()