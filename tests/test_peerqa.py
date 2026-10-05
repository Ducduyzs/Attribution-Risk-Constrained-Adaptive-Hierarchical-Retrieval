from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from edahr.baselines import auto_label_gold_children
from edahr.config import Settings
from edahr.hierarchy import HierarchyBuilder
from edahr.peerqa import convert_peerqa, mteb_to_peerqa_rows
from edahr.qasper import documents_from_paper_records

PID = "openreview/ICLR-2022-conf/abc"
ROWS = [
    {"idx": 0, "pidx": 0, "sidx": 0, "type": "title", "content": "A Paper", "last_heading": None, "paper_id": PID},
    {"idx": 1, "pidx": 1, "sidx": 0, "type": "heading", "content": "Abstract", "last_heading": None, "paper_id": PID},
    {"idx": 2, "pidx": 2, "sidx": 0, "type": "sentence", "content": "We study gold.", "last_heading": "Abstract", "paper_id": PID},
    {"idx": 3, "pidx": 2, "sidx": 1, "type": "sentence", "content": "It works well.", "last_heading": "Abstract", "paper_id": PID},
    {"idx": 4, "pidx": 3, "sidx": 0, "type": "heading", "content": "Results", "last_heading": "Abstract", "paper_id": PID},
    {"idx": 5, "pidx": 4, "sidx": 0, "type": "sentence", "content": "Accuracy is 91 percent.", "last_heading": "Results", "paper_id": PID},
    {"idx": 6, "pidx": 5, "sidx": 0, "type": "table", "content": "Table 1: scores.", "last_heading": "Results", "paper_id": PID},
]
QA = [
    {"paper_id": PID, "question_id": "q1", "question": "What is the accuracy?",
     "answer_free_form": "91 percent", "answerable": True, "answerable_mapped": True,
     "answer_evidence_mapped": [{"sentence": "Accuracy is 91 percent.", "idx": [5]}]},
    {"paper_id": PID, "question_id": "q2", "question": "Unanswerable?", "answer_free_form": "",
     "answerable": False, "answerable_mapped": False, "answer_evidence_mapped": []},
    {"paper_id": "missing", "question_id": "q3", "question": "x", "answerable_mapped": True,
     "answer_evidence_mapped": [{"idx": [1]}]},
]


class PeerQAConversionTests(unittest.TestCase):
    def test_structure_and_evidence(self):
        papers, questions, report = convert_peerqa(ROWS, QA)
        self.assertEqual(report["skipped"], {"not_answerable_mapped": 1, "paper_missing": 1,
                                             "evidence_unmapped": 0})
        paper = papers[0]
        self.assertEqual(paper["title"], "A Paper")
        self.assertEqual([s["title"] for s in paper["sections"]], ["Abstract", "Results"])
        abstract = paper["sections"][0]["paragraphs"][0]
        self.assertEqual(abstract["text"], "We study gold. It works well.")
        self.assertEqual(len(paper["sections"][1]["paragraphs"]), 2)  # sentence + table caption
        question = questions[0]
        self.assertEqual(question["reference_evidence_sets"], [["Accuracy is 91 percent."]])
        self.assertEqual(question["gold_paragraph_ids"], [f"{PID}:p4"])
        self.assertEqual(question["source"], "openreview_ICLR-2022-conf_abc.peerqa")

    def test_gold_leaves_resolve_in_pipeline_hierarchy(self):
        papers, questions, _ = convert_peerqa(ROWS, QA)
        hierarchy = HierarchyBuilder(Settings(child_target_tokens=50)).build(
            documents_from_paper_records(papers))
        gold, _ = auto_label_gold_children(hierarchy, questions[0])
        self.assertTrue(gold)
        self.assertTrue(all("91 percent" in hierarchy.node(c).text for c in gold))

    def test_mteb_packaging(self):
        corpus = [
            {"id": "nlpeer/X/1_0", "text": "A Title", "title": ""},
            {"id": "nlpeer/X/1_1", "text": "Abstract", "title": ""},
            {"id": "nlpeer/X/1_2", "text": "We study gold.", "title": "Abstract"},
            {"id": "nlpeer/X/1_3", "text": "Results", "title": ""},
            {"id": "nlpeer/X/1_4", "text": "Accuracy is 91 percent.", "title": "Results"},
            {"id": "nlpeer/X/1_5", "text": "Table 2: caption without heading.", "title": ""},
        ]
        rows, qa = mteb_to_peerqa_rows(corpus, [{"id": "q", "text": "Accuracy?"}, {"id": "none", "text": "x"}],
                                       [{"query-id": "q", "corpus-id": "nlpeer/X/1_4", "score": 1}])
        papers, questions, report = convert_peerqa(rows, qa)
        self.assertEqual(papers[0]["title"], "A Title")
        self.assertEqual([s["title"] for s in papers[0]["sections"]], ["Abstract", "Results"])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["reference_evidence_sets"], [["Accuracy is 91 percent."]])
        # An empty-heading unit that no unit names as heading stays body text.
        body = [p["text"] for s in papers[0]["sections"] for p in s["paragraphs"]]
        self.assertIn("Table 2: caption without heading.", body)


if __name__ == "__main__":
    unittest.main()
