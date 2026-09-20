"""Run one GPT-4o reader over leaf-normalized contexts from frozen systems.

This controlled reader experiment reuses the exact candidate leaf IDs saved by
the completed main run.  It therefore does not rerun retrieval and does not
mix post-generation verifier evidence into the reader input.  Results are
checkpointed after every successful request and API keys are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SYSTEMS = {
    "B3_flat_neural": "B3_flat_neural_rows.jsonl",
    "B4_static_hierarchy": "B4_static_hierarchy_rows.jsonl",
    "edahr_prior": "edahr_prior_rows.jsonl",
    "B5_raptor_faithful": "B5_raptor_faithful_rows.jsonl",
}


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


def build_prompt(query: str, passages: list[tuple[str, str]]) -> str:
    rendered = "\n\n".join(
        f"[{index}] passage_id={node_id}\n{text}"
        for index, (node_id, text) in enumerate(passages, start=1)
    )
    return (
        "Answer the scientific-document question using only the evidence passages. "
        "If the evidence is insufficient, answer Unanswerable. Give a concise answer; "
        "do not add outside knowledge.\n\n"
        f"Question: {query}\n\nEvidence passages:\n{rendered}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--systems", nargs="+", choices=tuple(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument("--max-pending", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    args = parser.parse_args()

    from openai import OpenAI

    from edahr.config import Settings
    from edahr.evaluation import qasper_answer_token_f1, qasper_evidence_f1
    from edahr.hierarchy import HierarchyBuilder
    from edahr.qasper import documents_from_paper_records

    config = Settings.from_json(PROJECT_ROOT / args.config)
    api_key = os.getenv("OPENAI_API_KEY") or config.openai_api_key
    if not api_key or api_key == "<redacted>":
        raise RuntimeError("Set OPENAI_API_KEY or use the private local config")

    main_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main"
    output_dir = main_dir / "shared_gpt4o_reader"
    papers = load_jsonl(PROJECT_ROOT / "manifests" / "qasper_baseline_dev_papers.jsonl")
    questions = load_jsonl(PROJECT_ROOT / "manifests" / "qasper_baseline_dev_questions.jsonl")
    question_by_id = {str(row["question_id"]): row for row in questions}
    hierarchy = HierarchyBuilder(config).build(documents_from_paper_records(papers))
    client = OpenAI(api_key=api_key, timeout=180.0, max_retries=5)

    for system in args.systems:
        source_rows = load_jsonl(main_dir / SYSTEMS[system])
        rows_path = output_dir / f"{system}_rows.jsonl"
        errors_path = output_dir / f"{system}_errors.jsonl"
        metadata_path = output_dir / f"{system}_metadata.json"
        completed = {
            str(row["question_id"]): row for row in load_jsonl(rows_path)
            if row.get("question_id") is not None
        }
        pending = [row for row in source_rows if str(row.get("question_id")) not in completed]
        if args.max_pending:
            pending = pending[: args.max_pending]
        errors: list[dict] = []
        print(f"{system}: {len(completed)} complete, {len(pending)} selected", flush=True)

        for index, source_row in enumerate(pending, start=1):
            question_id = str(source_row["question_id"])
            question = question_by_id[question_id]
            passages: list[tuple[str, str]] = []
            predicted_paragraphs: dict[str, str] = {}
            for node_id in source_row.get("candidate_child_ids") or []:
                node = hierarchy.node(str(node_id))
                passages.append((str(node_id), node.text))
                predicted_paragraphs.update(node.metadata.get("paragraph_texts") or {})
            prompt = build_prompt(str(question["query"]), passages)
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "Answer only from the supplied evidence."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=args.max_output_tokens,
                    temperature=0,
                    seed=args.seed,
                )
                answer = (response.choices[0].message.content or "").strip()
                if not answer:
                    raise RuntimeError("OpenAI returned an empty answer")
                usage = response.usage
                row = {
                    "system": system,
                    "question_id": question_id,
                    "source": question.get("source") or question.get("paper_id"),
                    "query": question["query"],
                    "answer": answer,
                    "answer_f1": qasper_answer_token_f1(
                        answer, [str(item) for item in question.get("reference_answers") or [""]]
                    ),
                    "official_qasper_evidence_f1": qasper_evidence_f1(
                        list(predicted_paragraphs.values()),
                        question.get("reference_evidence_sets") or [],
                    ),
                    "candidate_child_ids": source_row.get("candidate_child_ids") or [],
                    "requested_model": args.model,
                    "response_model": response.model,
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
                completed[question_id] = row
                order = {str(item["question_id"]): i for i, item in enumerate(questions)}
                write_jsonl(
                    sorted(completed.values(), key=lambda item: order[str(item["question_id"])]),
                    rows_path,
                )
                print(
                    f"{system} checkpoint {index}/{len(pending)}: {question_id} "
                    f"tokens={row['total_tokens']}", flush=True,
                )
            except Exception:  # noqa: BLE001
                errors.append({
                    "system": system, "question_id": question_id,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "error": traceback.format_exc(),
                })
                print(f"{system} FAILED {question_id}", flush=True)

        write_jsonl(errors, errors_path)
        final_rows = load_jsonl(rows_path)
        metadata = {
            "task": "shared GPT-4o reader over leaf-normalized frozen contexts",
            "system": system,
            "requested_model": args.model,
            "seed": args.seed,
            "temperature": 0,
            "questions_total": len(source_rows),
            "rows_after": len(final_rows),
            "errors_this_attempt": len(errors),
            "prompt_tokens": sum(row["prompt_tokens"] for row in final_rows),
            "completion_tokens": sum(row["completion_tokens"] for row in final_rows),
            "total_tokens": sum(row["total_tokens"] for row in final_rows),
            "reader_input": "candidate_child_ids from frozen main rows, rendered as leaf text",
            "api_key_serialized": False,
        }
        metadata_path.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
        print(f"{system}: rows={len(final_rows)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
