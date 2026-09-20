"""Resume the interrupted LongRAG local-reader phase without redoing valid rows.

This recovery runner deliberately separates semantic retrieval from generation:
the BGE query encoder is released before Qwen is loaded, keeping the two models
from competing for VRAM on a 24-GB GPU.  Each successful answer is checkpointed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact_secrets(value):
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            lowered = str(key).lower()
            sensitive = (
                lowered in {
                    "api_key", "apikey", "access_token", "refresh_token",
                    "auth_token", "bearer_token", "password", "secret",
                    "client_secret", "credential", "credentials",
                }
                or lowered.endswith(("_api_key", "_password", "_secret"))
            )
            if sensitive:
                clean[key] = "<redacted>" if item else None
            else:
                clean[key] = redact_secrets(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pending", type=int, default=0)
    args = parser.parse_args()

    import torch

    from edahr import baselines_longrag as longrag_module
    from edahr.baselines import make_baseline_pipeline
    from edahr.baselines_longrag import LocalFreeReader, build_document_units
    from edahr.config import Settings
    from edahr.evaluation import qasper_answer_token_f1, qasper_evidence_f1
    from edahr.hierarchy import HierarchyBuilder
    from edahr.qasper import documents_from_paper_records

    manifest_dir = PROJECT_ROOT / "manifests"
    output_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main"
    rows_path = output_dir / "B6_longrag_reader_rows.jsonl"
    errors_path = output_dir / "errors_reader_resume.jsonl"
    metadata_path = output_dir / "run_metadata_reader_resume.json"
    questions_path = manifest_dir / "qasper_baseline_dev_questions.jsonl"
    papers_path = manifest_dir / "qasper_baseline_dev_papers.jsonl"

    settings = replace(
        Settings.from_json(PROJECT_ROOT / args.config), seed=args.seed)
    questions = load_jsonl(questions_path)
    papers = load_jsonl(papers_path)
    hierarchy = HierarchyBuilder(settings).build(
        documents_from_paper_records(papers))
    units = {unit.unit_id: unit for unit in build_document_units(hierarchy)}

    # No reranker, verifier, or shared generator is needed for retrieval traces.
    pipeline = make_baseline_pipeline(
        "B6_longrag_faithful", hierarchy, reranker=None, generator=None,
        verifier=None, settings=settings)

    existing = load_jsonl(rows_path)
    rows_by_id = {
        str(row["question_id"]): row for row in existing
        if row.get("question_id") is not None
    }
    pending = [
        question for question in questions
        if str(question.get("question_id")) not in rows_by_id
    ]
    if args.max_pending:
        pending = pending[:args.max_pending]
    print(
        f"reader resume: {len(rows_by_id)} existing, {len(pending)} selected ",
        f"of {len(questions) - len(rows_by_id)} pending",
        flush=True,
    )

    errors: list[dict] = []
    work: list[tuple[dict, list]] = []
    for question in pending:
        try:
            trace = pipeline.retriever.retrieval_trace(
                question["query"], question.get("source"))
            picked = [
                units[item["unit_id"]] for item in trace["picked"]
                if item["unit_id"] in units
            ]
            work.append((question, picked))
        except Exception:  # noqa: BLE001
            errors.append({
                "where": "reader_retrieval",
                "question_id": question.get("question_id"),
                "error": traceback.format_exc(),
            })

    # FlagEmbedding caches the GPU model at module scope. Release it before Qwen.
    longrag_module._FLAG_MODEL_CACHE.clear()
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    reader = LocalFreeReader("Qwen/Qwen2.5-7B-Instruct", settings.device)
    question_order = {
        str(question.get("question_id")): index
        for index, question in enumerate(questions)
    }

    def ordered_rows() -> list[dict]:
        return sorted(
            rows_by_id.values(),
            key=lambda row: question_order.get(
                str(row.get("question_id")), len(question_order)),
        )

    for index, (question, picked) in enumerate(work, start=1):
        question_id = str(question.get("question_id"))
        try:
            answer = reader.generate_answer(question["query"], picked)
            references = [
                str(answer) for answer in
                (question.get("reference_answers") or [""])
            ]
            rows_by_id[question_id] = {
                "question_id": question.get("question_id"),
                "answer": answer,
                "answer_f1": qasper_answer_token_f1(
                    answer or "Unanswerable", references),
                "evidence_f1": qasper_evidence_f1(
                    [text for unit in picked for _, _, _, text in unit.paragraphs],
                    question.get("reference_evidence_sets") or [],
                ),
                "picked_units": [unit.unit_id for unit in picked],
            }
            write_jsonl(ordered_rows(), rows_path)
            print(
                f"checkpoint {index}/{len(work)}: {question_id} ",
                f"({len(rows_by_id)}/{len(questions)} complete)",
                flush=True,
            )
        except Exception:  # noqa: BLE001
            errors.append({
                "where": "reader_row", "question_id": question_id,
                "error": traceback.format_exc(),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"failed {question_id}", flush=True)

    write_jsonl(ordered_rows(), rows_path)
    write_jsonl(errors, errors_path)
    peak_gb = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available() else 0.0
    )
    metadata = {
        "task": "resume LongRAG local reader",
        "seed": args.seed,
        "questions_total": len(questions),
        "rows_before": len(existing),
        "rows_after": len(rows_by_id),
        "selected_pending": len(pending),
        "errors": len(errors),
        "gpu_peak_gb": peak_gb,
        "manifest_hashes": {
            "dev_papers": sha256_file(papers_path),
            "dev_questions": sha256_file(questions_path),
        },
        "settings": redact_secrets(settings.to_dict()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "rows_before", "rows_after", "errors", "gpu_peak_gb")}), flush=True)


if __name__ == "__main__":
    main()
