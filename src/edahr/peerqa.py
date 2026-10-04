"""PeerQA -> the paper/question record format produced by ``convert_qasper``.

PeerQA (Baumgaertner et al., NAACL 2025) ships ``papers.jsonl`` as one row per
GROBID element (title, heading, sentence, list_item, formula, figure, table)
with a per-paper element index ``idx``, paragraph index ``pidx`` and the last
heading, and ``qa.jsonl`` whose ``answer_evidence_mapped`` points at element
``idx`` values. Here elements are grouped into paragraphs (by ``pidx``) and
sections (by heading), keeping stable paragraph ids and character offsets so
the rest of the pipeline (leaf labelling, paragraph evidence F1) is unchanged.

Only ``answerable_mapped`` questions have gold evidence; the evidence of the
single author annotation becomes one reference paragraph set.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

BODY_TYPES = {"sentence", "list_item", "formula", "figure", "table"}


def _norm(text: object) -> str:
    return " ".join(str(text or "").split())


def safe_source(paper_id: str) -> str:
    return paper_id.replace("/", "_") + ".peerqa"


def convert_peerqa(paper_rows: Iterable[dict], qa_rows: Iterable[dict]) -> tuple[list[dict], list[dict], dict]:
    by_paper: dict[str, list[dict]] = OrderedDict()
    for row in paper_rows:
        by_paper.setdefault(str(row["paper_id"]), []).append(row)

    papers: list[dict] = []
    element_to_paragraph: dict[tuple[str, int], str] = {}
    paragraph_text: dict[str, str] = {}
    for paper_id, rows in by_paper.items():
        rows = sorted(rows, key=lambda r: int(r["idx"]))
        title = next((_norm(r["content"]) for r in rows if r.get("type") == "title"), "")
        sections: list[dict] = []
        current: dict | None = None
        paragraphs: "OrderedDict[str, list[dict]]" = OrderedDict()

        def flush_section():
            if current is None or not paragraphs:
                return
            metadata, cursor, texts = [], 0, []
            for pid, items in paragraphs.items():
                text = _norm(" ".join(_norm(i["content"]) for i in items))
                if not text:
                    continue
                metadata.append({"paragraph_id": pid, "text": text,
                                 "char_start": cursor, "char_end": cursor + len(text)})
                paragraph_text[pid] = text
                texts.append(text)
                cursor += len(text) + 1
            if texts:
                current["paragraphs"] = metadata
                current["text"] = "\n".join(texts)
                sections.append(current)

        for row in rows:
            kind = row.get("type")
            if kind == "heading" or current is None:
                flush_section()
                heading = _norm(row["content"]) if kind == "heading" else _norm(row.get("last_heading")) or "Preamble"
                current = {"title": heading, "section_type": "document",
                           "position": len(sections)}
                paragraphs = OrderedDict()
                if kind == "heading":
                    continue
            if kind not in BODY_TYPES:
                continue
            pid = f"{paper_id}:p{row['pidx']}"
            paragraphs.setdefault(pid, []).append(row)
            element_to_paragraph[(paper_id, int(row["idx"]))] = pid
        flush_section()
        papers.append({"dataset": "peerqa", "split": "test", "paper_id": paper_id,
                       "document_id": safe_source(paper_id)[:-len(".peerqa")],
                       "source": safe_source(paper_id), "title": title, "sections": sections})

    known = {p["paper_id"] for p in papers}
    questions: list[dict] = []
    skipped = {"not_answerable_mapped": 0, "paper_missing": 0, "evidence_unmapped": 0}
    for qa in qa_rows:
        if not qa.get("answerable_mapped"):
            skipped["not_answerable_mapped"] += 1
            continue
        paper_id = str(qa["paper_id"])
        if paper_id not in known:
            skipped["paper_missing"] += 1
            continue
        pids: list[str] = []
        for item in qa.get("answer_evidence_mapped") or ():
            for idx in item.get("idx") or ():
                pid = element_to_paragraph.get((paper_id, int(idx)))
                if pid and pid not in pids:
                    pids.append(pid)
        if not pids:
            skipped["evidence_unmapped"] += 1
            continue
        texts = [paragraph_text[pid] for pid in pids]
        questions.append({
            "dataset": "peerqa", "split": "test", "paper_id": paper_id,
            "source": safe_source(paper_id), "question_id": str(qa["question_id"]),
            "query": _norm(qa["question"]), "answer": _norm(qa.get("answer_free_form")),
            "reference_answers": [_norm(qa.get("answer_free_form"))],
            "reference_evidence_sets": [texts], "reference_paragraph_sets": [pids],
            "gold_paragraph_ids": sorted(pids), "gold_quotes": texts,
            "citation_evaluable_source": True,
        })
    report = {"papers": len(papers), "questions": len(questions), "skipped": skipped,
              "paragraphs": len(paragraph_text)}
    return papers, questions, report
