"""Run the reported LongRAG GPT-4o reader on the frozen dev manifest.

The script checkpoints each successful query and never serializes the API key.
Retrieval and reader telemetry are retained per query for reproducibility.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def usage_value(usage, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return int(value or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-pending", type=int, default=0)
    args = parser.parse_args()

    import torch
    from openai import OpenAI

    from edahr import baselines_longrag as longrag_module
    from edahr.baselines import make_baseline_pipeline
    from edahr.baselines_longrag import build_document_units, build_reader_prompt
    from edahr.config import Settings
    from edahr.evaluation import qasper_answer_token_f1, qasper_evidence_f1
    from edahr.hierarchy import HierarchyBuilder
    from edahr.qasper import documents_from_paper_records

    manifest_dir = PROJECT_ROOT / "manifests"
    output_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main"
    questions_path = manifest_dir / "qasper_baseline_dev_questions.jsonl"
    papers_path = manifest_dir / "qasper_baseline_dev_papers.jsonl"
    manifest_metadata_path = manifest_dir / "qasper_baseline_manifest_metadata.json"
    rows_path = output_dir / "B6_longrag_reader_gpt4o_rows.jsonl"
    traces_path = output_dir / "B6_longrag_reader_gpt4o_traces.jsonl"
    errors_path = output_dir / "errors_reader_gpt4o.jsonl"
    metadata_path = output_dir / "run_metadata_reader_gpt4o.json"

    settings = replace(
        Settings.from_json(PROJECT_ROOT / args.config), seed=args.seed)
    api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key or api_key == "<redacted>":
        raise RuntimeError(
            "Set OPENAI_API_KEY or openai_api_key in the private runtime config")

    questions = load_jsonl(questions_path)
    papers = load_jsonl(papers_path)
    hierarchy = HierarchyBuilder(settings).build(
        documents_from_paper_records(papers))
    units = {unit.unit_id: unit for unit in build_document_units(hierarchy)}
    pipeline = make_baseline_pipeline(
        "B6_longrag_faithful", hierarchy, reranker=None, generator=None,
        verifier=None, settings=settings)

    rows_by_id = {
        str(row["question_id"]): row for row in load_jsonl(rows_path)
        if row.get("question_id") is not None
    }
    traces_by_id = {
        str(row["question_id"]): row for row in load_jsonl(traces_path)
        if row.get("question_id") is not None
    }
    pending = [
        question for question in questions
        if str(question.get("question_id")) not in rows_by_id
    ]
    pending_total = len(pending)
    if args.max_pending:
        pending = pending[:args.max_pending]
    print(
        f"gpt-4o reader: {len(rows_by_id)} existing, {len(pending)} selected "
        f"of {pending_total} pending", flush=True)

    errors: list[dict] = []
    work: list[tuple[dict, list, dict]] = []
    for question in pending:
        question_id = str(question.get("question_id"))
        try:
            trace = pipeline.retriever.retrieval_trace(
                question["query"], question.get("source"))
            picked = [
                units[item["unit_id"]] for item in trace["picked"]
                if item["unit_id"] in units
            ]
            work.append((question, picked, trace))
        except Exception:  # noqa: BLE001
            errors.append({
                "where": "reader_retrieval", "question_id": question_id,
                "error": traceback.format_exc(),
            })

    longrag_module._FLAG_MODEL_CACHE.clear()
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    client = OpenAI(api_key=api_key, timeout=180.0, max_retries=5)
    question_order = {
        str(question["question_id"]): index
        for index, question in enumerate(questions)
    }

    def ordered(mapping: dict[str, dict]) -> list[dict]:
        return sorted(
            mapping.values(),
            key=lambda row: question_order.get(
                str(row.get("question_id")), len(question_order)),
        )

    for index, (question, picked, retrieval_trace) in enumerate(work, start=1):
        question_id = str(question.get("question_id"))
        prompt = build_reader_prompt(question["query"], picked)
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer concisely using only the supplied units.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=args.max_output_tokens,
                temperature=0,
                seed=args.seed,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            choice = response.choices[0]
            answer = (choice.message.content or "").strip()
            if not answer:
                raise RuntimeError("OpenAI returned an empty answer")
            references = [
                str(answer) for answer in
                (question.get("reference_answers") or [""])
            ]
            usage = response.usage
            cached_tokens = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_tokens = int(
                    getattr(prompt_details, "cached_tokens", 0) or 0)
            row = {
                "question_id": question.get("question_id"),
                "source": question.get("source") or question.get("paper_id"),
                "query": question.get("query"),
                "answer": answer,
                "answer_f1": qasper_answer_token_f1(
                    answer or "Unanswerable", references),
                "official_qasper_evidence_f1": qasper_evidence_f1(
                    [text for unit in picked for _, _, _, text in unit.paragraphs],
                    question.get("reference_evidence_sets") or [],
                ),
                "picked_units": [unit.unit_id for unit in picked],
                "requested_model": args.model,
                "response_model": response.model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": choice.finish_reason,
                "prompt_tokens": usage_value(usage, "prompt_tokens"),
                "cached_prompt_tokens": cached_tokens,
                "completion_tokens": usage_value(usage, "completion_tokens"),
                "total_tokens": usage_value(usage, "total_tokens"),
                "latency_ms": latency_ms,
                "prompt_sha256": sha256_text(prompt),
            }
            rows_by_id[question_id] = row
            traces_by_id[question_id] = {
                "question_id": question.get("question_id"),
                "source": row["source"],
                "retrieval": retrieval_trace,
                "picked_units": row["picked_units"],
                "prompt_sha256": row["prompt_sha256"],
                "prompt_tokens": row["prompt_tokens"],
                "finish_reason": row["finish_reason"],
            }
            write_jsonl(ordered(rows_by_id), rows_path)
            write_jsonl(ordered(traces_by_id), traces_path)
            print(
                f"checkpoint {index}/{len(work)}: {question_id} "
                f"({len(rows_by_id)}/{len(questions)} complete; "
                f"tokens={row['total_tokens']})", flush=True)
        except Exception:  # noqa: BLE001
            errors.append({
                "where": "reader_api", "question_id": question_id,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "error": traceback.format_exc(),
            })
            print(f"failed {question_id}", flush=True)

    final_rows = ordered(rows_by_id)
    write_jsonl(final_rows, rows_path)
    write_jsonl(ordered(traces_by_id), traces_path)
    write_jsonl(errors, errors_path)
    metadata = {
        "task": "LongRAG reported GPT-4o reader on frozen dev manifest",
        "reader_provider": "openai",
        "requested_model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "max_output_tokens": args.max_output_tokens,
        "questions_total": len(questions),
        "rows_after": len(final_rows),
        "selected_pending": len(pending),
        "errors_this_attempt": len(errors),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in final_rows),
        "cached_prompt_tokens": sum(
            int(row.get("cached_prompt_tokens", 0)) for row in final_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens", 0)) for row in final_rows),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in final_rows),
        "total_latency_ms": sum(
            float(row.get("latency_ms", 0.0)) for row in final_rows),
        "response_models": sorted({
            str(row.get("response_model")) for row in final_rows
            if row.get("response_model")
        }),
        "system_fingerprints": sorted({
            str(row.get("system_fingerprint")) for row in final_rows
            if row.get("system_fingerprint")
        }),
        "manifest_hashes": {
            "dev_papers": sha256_file(papers_path),
            "dev_questions": sha256_file(questions_path),
            "manifest_metadata": sha256_file(manifest_metadata_path),
        },
        "predictions_sha256": sha256_file(rows_path) if final_rows else None,
        "credentials_serialized": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in (
        "rows_after", "errors_this_attempt", "prompt_tokens",
        "completion_tokens", "total_tokens", "response_models",
    )}), flush=True)


if __name__ == "__main__":
    main()
