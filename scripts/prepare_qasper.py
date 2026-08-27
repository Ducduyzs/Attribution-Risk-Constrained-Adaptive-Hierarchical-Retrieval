"""Convert official QASPER v0.3 files into reproducible JSONL manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.qasper import convert_qasper, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasper-dir", default="data/qasper")
    parser.add_argument("--out-dir", default="data/manifests")
    args = parser.parse_args()
    qasper_dir = Path(args.qasper_dir)
    out_dir = Path(args.out_dir)
    if not qasper_dir.is_absolute():
        qasper_dir = PROJECT_ROOT / qasper_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    reports: list[dict] = []
    question_ids: dict[str, str] = {}
    paper_ids: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        raw = qasper_dir / f"qasper-{split}-v0.3.json"
        papers, questions, report = convert_qasper(raw, split)
        for row in questions:
            previous = question_ids.setdefault(row["question_id"], split)
            if previous != split:
                raise ValueError(f"question leakage: {row['question_id']} in {previous}/{split}")
        for row in papers:
            previous = paper_ids.setdefault(row["paper_id"], split)
            if previous != split:
                raise ValueError(f"paper leakage: {row['paper_id']} in {previous}/{split}")
        write_jsonl(papers, out_dir / f"qasper_{split}_papers.jsonl")
        write_jsonl(questions, out_dir / f"qasper_{split}_questions.jsonl")
        reports.append(report)

    summary = {
        "schema_version": 1, "dataset": "qasper-v0.3",
        "splits": reports,
        "total_papers": len(paper_ids), "total_questions": len(question_ids),
        "paper_overlap_across_splits": 0,
        "question_overlap_across_splits": 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "qasper_manifest_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
