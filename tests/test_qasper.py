from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.qasper import convert_qasper, documents_from_paper_records


class QasperConversionTests(unittest.TestCase):
    def test_stable_identity_references_and_evidence_union(self):
        raw = {
            "paper-1": {
                "title": "A paper",
                "abstract": "Abstract evidence.",
                "full_text": [{
                    "section_name": "Results",
                    "paragraphs": ["Gold paragraph.", "Other paragraph."],
                }],
                "qas": [{
                    "question": "What was found?",
                    "question_id": "stable-q1",
                    "answers": [
                        {"answer": {
                            "unanswerable": False,
                            "free_form_answer": "First answer",
                            "extractive_spans": [],
                            "yes_no": None,
                            "evidence": ["Gold paragraph."],
                            "highlighted_evidence": [],
                        }},
                        {"answer": {
                            "unanswerable": False,
                            "free_form_answer": "Second answer",
                            "extractive_spans": [],
                            "yes_no": None,
                            "evidence": ["Gold paragraph.", "Other paragraph."],
                            "highlighted_evidence": [],
                        }},
                    ],
                }],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            papers, questions, report = convert_qasper(path, "train")

        self.assertEqual(report["papers"], 1)
        self.assertEqual(report["stable_question_ids"], 1)
        self.assertEqual(questions[0]["question_id"], "stable-q1")
        self.assertEqual(questions[0]["source"], "paper-1.qasper")
        self.assertEqual(
            questions[0]["reference_answers"], ["First answer", "Second answer"]
        )
        self.assertEqual(
            questions[0]["gold_quotes"], ["Gold paragraph.", "Other paragraph."]
        )
        documents = documents_from_paper_records(papers)
        self.assertEqual(documents[0].source, questions[0]["source"])
        self.assertTrue(any(section.title == "Abstract" for section in documents[0].sections))

    def test_duplicate_question_id_is_rejected(self):
        qa = {"question": "q", "question_id": "same", "answers": []}
        raw = {
            "paper": {
                "title": "p", "abstract": "a",
                "full_text": [{"section_name": "S", "paragraphs": ["x"]}],
                "qas": [qa, qa],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate question_id"):
                convert_qasper(path, "train")


if __name__ == "__main__":
    unittest.main()
