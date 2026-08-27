"""Deterministic QASPER v0.3 conversion with stable paper/question identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from .schemas import DocumentSection, ScientificDocument


def _unique_text(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _answer_text(answer: dict) -> str:
    if answer.get("unanswerable"):
        return ""
    free_form = str(answer.get("free_form_answer") or "").strip()
    if free_form:
        return free_form
    spans = _unique_text(answer.get("extractive_spans") or ())
    if spans:
        return " ".join(spans)
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "yes" if yes_no else "no"
    if str(yes_no or "").strip().lower() in {"yes", "no"}:
        return str(yes_no).strip().lower()
    return ""


def convert_qasper(raw_path: str | Path, split: str) -> tuple[list[dict], list[dict], dict]:
    """Convert one official QASPER JSON file into paper and question records.

    Evidence is the deterministic union of annotator evidence. Reference
    answers remain separate so answer F1 can use max-over-references.
    """
    source_path = Path(raw_path)
    raw = json.loads(source_path.read_text(encoding="utf-8-sig"))
    papers: list[dict] = []
    questions: list[dict] = []
    seen_question_ids: set[str] = set()

    for paper_id, paper in raw.items():
        source = f"{paper_id}.qasper"
        sections = []
        for position, section in enumerate(paper.get("full_text") or ()):
            paragraphs = _unique_text(section.get("paragraphs") or ())
            text = "\n".join(paragraphs).strip()
            if not text:
                continue
            sections.append({
                "title": str(section.get("section_name") or f"Section {position + 1}"),
                "text": text,
                "section_type": "document",
                "position": position,
            })
        abstract = str(paper.get("abstract") or "").strip()
        if abstract and not any(section["title"].casefold() == "abstract" for section in sections):
            sections.insert(0, {
                "title": "Abstract", "text": abstract,
                "section_type": "abstract", "position": -1,
            })
        papers.append({
            "dataset": "qasper", "split": split, "paper_id": paper_id,
            "document_id": paper_id, "source": source,
            "title": str(paper.get("title") or ""), "sections": sections,
        })

        for qa in paper.get("qas") or ():
            question_id = str(qa.get("question_id") or "")
            if not question_id:
                raise ValueError(f"QASPER question without question_id in paper {paper_id}")
            if question_id in seen_question_ids:
                raise ValueError(f"duplicate question_id in {split}: {question_id}")
            seen_question_ids.add(question_id)
            annotations = [item.get("answer") or {} for item in (qa.get("answers") or ())]
            references = _unique_text(_answer_text(answer) for answer in annotations)
            evidence = _unique_text(
                quote
                for answer in annotations
                for quote in [
                    *(answer.get("evidence") or ()),
                    *(answer.get("highlighted_evidence") or ()),
                ]
            )
            questions.append({
                "dataset": "qasper", "split": split,
                "paper_id": paper_id, "source": source,
                "question_id": question_id,
                "query": str(qa.get("question") or "").strip(),
                "answer": references[0] if references else "",
                "reference_answers": references,
                "gold_quotes": evidence,
                "citation_evaluable_source": bool(evidence),
                "all_annotations_unanswerable": bool(annotations) and all(
                    bool(answer.get("unanswerable")) for answer in annotations
                ),
                "annotation_count": len(annotations),
            })

    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    report = {
        "dataset": "qasper", "split": split, "source": str(source_path.resolve()),
        "source_sha256": digest, "papers": len(papers), "questions": len(questions),
        "stable_question_ids": len(seen_question_ids),
        "questions_with_gold_quotes": sum(bool(row["gold_quotes"]) for row in questions),
        "questions_with_reference_answers": sum(bool(row["reference_answers"]) for row in questions),
        "all_annotations_unanswerable": sum(
            bool(row["all_annotations_unanswerable"]) for row in questions
        ),
    }
    report["gold_quote_question_rate"] = (
        report["questions_with_gold_quotes"] / max(1, report["questions"])
    )
    return papers, questions, report


def write_jsonl(rows: Sequence[dict], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def documents_from_paper_records(records: Sequence[dict]) -> list[ScientificDocument]:
    documents: list[ScientificDocument] = []
    for record in records:
        sections = tuple(
            DocumentSection(
                title=str(section["title"]), text=str(section["text"]),
                section_type=str(section.get("section_type") or "document"),
                metadata={"qasper_position": section.get("position")},
            )
            for section in record.get("sections") or ()
            if str(section.get("text") or "").strip()
        )
        documents.append(ScientificDocument(
            document_id=str(record["document_id"]),
            source=str(record["source"]), sections=sections,
            metadata={
                "dataset": record.get("dataset") or "qasper", "split": record.get("split"),
                "title": record.get("title"), "paper_id": record.get("paper_id"),
            },
        ))
    return documents
