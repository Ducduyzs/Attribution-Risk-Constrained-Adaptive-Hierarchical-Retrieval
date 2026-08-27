"""Run fresh paper-scoped rollouts from prepared QASPER manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import auto_label_gold_children  # noqa: E402
from edahr.config import Settings  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl, write_jsonl  # noqa: E402
from edahr.rollouts import RolloutRunner  # noqa: E402
from edahr.runtime import build_pipeline_from_documents  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_records(
    papers: list[dict], questions: list[dict], *,
    paper_limit: int | None, question_limit: int | None,
    citation_only: bool, questions_per_paper: int,
) -> tuple[list[dict], list[dict]]:
    if citation_only:
        questions = [row for row in questions if row.get("gold_quotes")]
    allowed_papers: list[str] = []
    seen: set[str] = set()
    for row in questions:
        paper_id = str(row["paper_id"])
        if paper_id not in seen:
            seen.add(paper_id)
            allowed_papers.append(paper_id)
        if paper_limit is not None and len(allowed_papers) >= paper_limit:
            break
    allowed = set(allowed_papers)
    selected_questions: list[dict] = []
    counts: dict[str, int] = {}
    for row in questions:
        paper_id = str(row["paper_id"])
        if paper_id not in allowed or counts.get(paper_id, 0) >= questions_per_paper:
            continue
        selected_questions.append(row)
        counts[paper_id] = counts.get(paper_id, 0) + 1
    if question_limit is not None:
        selected_questions = selected_questions[:question_limit]
    used = {str(row["paper_id"]) for row in selected_questions}
    selected_papers = [row for row in papers if str(row["paper_id"]) in used]
    return selected_papers, selected_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--provider", choices=("openai", "gemini", "antigravity"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--paper-limit", type=int, default=None)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--questions-per-paper", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="append and skip question IDs already in --out")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=4)
    parser.add_argument("--include-non-evaluable", action="store_true")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = PROJECT_ROOT / manifest_dir
    paper_path = manifest_dir / f"qasper_{args.split}_papers.jsonl"
    question_path = manifest_dir / f"qasper_{args.split}_questions.jsonl"
    papers, questions = select_records(
        read_jsonl(paper_path), read_jsonl(question_path),
        paper_limit=args.paper_limit, question_limit=args.question_limit,
        citation_only=not args.include_non_evaluable,
        questions_per_paper=max(1, args.questions_per_paper),
    )
    if not papers or not questions:
        raise ValueError("selection produced no QASPER papers/questions")
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    completed_ids: set[str] = set()
    if args.resume and out.is_file():
        completed_ids = {
            str(row.get("question_id") or "") for row in read_jsonl(out)
        }
        questions = [
            row for row in questions if str(row["question_id"]) not in completed_ids
        ]
        used = {str(row["paper_id"]) for row in questions}
        papers = [row for row in papers if str(row["paper_id"]) in used]
        if not questions:
            print(json.dumps({"output": str(out), "status": "already-complete",
                              "completed_question_ids": len(completed_ids)}, indent=2))
            return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    settings = Settings.from_json(config_path)
    changes = {}
    if args.provider:
        changes["llm_provider"] = args.provider
    if args.model:
        changes["llm_model"] = args.model
    settings = replace(settings, **changes)

    documents = documents_from_paper_records(papers)
    pipeline = build_pipeline_from_documents(documents, settings)
    enriched: list[dict] = []
    for record in questions:
        copy = dict(record)
        gold_ids, _ = auto_label_gold_children(pipeline.hierarchy, copy)
        copy["gold_child_ids"] = sorted(gold_ids)
        copy["citation_evaluable"] = bool(gold_ids)
        enriched.append(copy)

    selected_path = out.with_suffix(".selection.jsonl")
    write_jsonl(enriched, selected_path)
    rows = RolloutRunner(
        pipeline, settings=settings, samples=args.samples,
        max_groups_per_query=args.max_groups,
    ).run_records(enriched, out, append=args.resume)
    evaluable_questions = sum(bool(row.get("citation_evaluable")) for row in enriched)
    all_rows = read_jsonl(out)
    metadata = {
        "schema_version": 1, "dataset": "qasper-v0.3", "split": args.split,
        "provider": settings.llm_provider, "model": settings.llm_model,
        "paper_count": len(papers), "question_count": len(enriched),
        "new_rollout_rows": len(rows), "total_rollout_rows": len(all_rows),
        "resumed_completed_question_ids": len(completed_ids),
        "citation_evaluable_questions": evaluable_questions,
        "citation_evaluable_rate": evaluable_questions / max(1, len(enriched)),
        "paper_manifest_sha256": sha256(paper_path),
        "question_manifest_sha256": sha256(question_path),
        "selection_sha256": sha256(selected_path),
        "settings": {
            key: value for key, value in settings.to_dict().items()
            if "api_key" not in key
        },
    }
    metadata_path = out.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
