"""Faithful-baseline smoke test on the frozen dev manifest (Task 8).

Runs B5_raptor_faithful and B6_longrag_faithful (controlled configuration:
faithful retrieval + shared repo generator stack) over the first --papers
papers of manifests/qasper_baseline_dev_*.jsonl, plus the LongRAG primary
long-context reader pass. Writes per-baseline run directories:

  artifacts/baselines/{raptor,longrag}/smoke/
    config.json run_metadata.json per_query_predictions.jsonl
    retrieval_traces.jsonl summary_metrics.json execution.log
    index_metadata.json errors.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import make_baseline_pipeline, run_benchmark  # noqa: E402
from edahr.config import Settings  # noqa: E402
from edahr.evaluation import qasper_answer_token_f1, qasper_evidence_f1  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import documents_from_paper_records  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], capture_output=True, text=True,
                cwd=PROJECT_ROOT,
            ).stdout.strip()
        )
        return {"git_commit": commit, "dirty_worktree": dirty}
    except Exception:
        return {"git_commit": "unknown", "dirty_worktree": True}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class _RetryAnswerWrapper:
    """Retry transient LLM API errors per query (503/429); failures recorded.

    Delegates everything to the wrapped pipeline; run_benchmark only uses
    .answer/.hierarchy/.settings/.retriever.
    """

    def __init__(self, pipeline, attempts: int = 6):
        self._pipeline = pipeline
        self._attempts = attempts
        self.retries = 0
        self.hierarchy = pipeline.hierarchy
        self.settings = pipeline.settings
        self.retriever = pipeline.retriever

    def answer(self, query: str, source=None):
        import time as _time

        last = None
        for attempt in range(1, self._attempts + 1):
            try:
                return self._pipeline.answer(query, source=source)
            except Exception as exc:  # noqa: BLE001 - transient API errors
                last = exc
                self.retries += 1
                _time.sleep(min(2**attempt, 60))
        raise last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config.local.json")
    parser.add_argument("--skip-raptor", action="store_true")
    parser.add_argument("--skip-longrag", action="store_true")
    args = parser.parse_args()

    log_lines: list[str] = []

    def log(message: str) -> None:
        stamped = f"{datetime.datetime.now().isoformat()} {message}"
        log_lines.append(stamped)
        print(stamped, flush=True)

    manifest_dir = PROJECT_ROOT / "manifests"
    dev_papers_path = manifest_dir / "qasper_baseline_dev_papers.jsonl"
    dev_questions_path = manifest_dir / "qasper_baseline_dev_questions.jsonl"
    metadata_path = manifest_dir / "qasper_baseline_manifest_metadata.json"
    settings = Settings.from_json(PROJECT_ROOT / args.config)
    settings = replace(settings, seed=args.seed)

    all_papers = load_jsonl(dev_papers_path)
    all_questions = load_jsonl(dev_questions_path)
    papers = all_papers[: args.papers]
    paper_ids = {str(p["paper_id"]) for p in papers}
    questions = [q for q in all_questions if str(q["paper_id"]) in paper_ids]
    log(f"smoke papers={len(papers)} questions={len(questions)}")

    multi_evidence = sum(1 for q in questions if len(q.get("gold_paragraph_ids", [])) > 1)
    log(f"questions with >1 evidence paragraphs: {multi_evidence}")

    from edahr.models import (  # noqa: E402
        BGEReranker,
        LocalStructuredGenerator,
        NliVerifier,
    )

    log("loading shared stack (bge-m3 encoder/reranker, NLI verifier, generator)...")
    # Lazy encoder: faithful RAPTOR/LongRAG branches never call index_factory,
    # so skip the ~2GB BGE-M3 download unless a flat baseline needs it.
    _encoder = None

    def index_factory(variant: Settings):
        nonlocal _encoder
        from edahr.index import MultiRepresentationIndex
        from edahr.models import BGEM3Encoder

        if _encoder is None:
            _encoder = BGEM3Encoder(
                settings.embedding_model, settings.device, settings.use_fp16
            )
        return MultiRepresentationIndex(hierarchy, _encoder, variant)

    reranker = BGEReranker(settings.reranker_model, settings.device)
    verifier = NliVerifier(settings.nli_model, settings.device)
    # API quotas exhausted on 2026-09-19 (OpenAI credits; Gemini free-tier
    # 20/day on every flash model tried). The shared controlled generator is
    # a local instruction-tuned LLM; recorded in metadata (adaptation B2/LG).
    # Qwen-3B showed 0% citation compliance (empty citations on all claims);
    # 7B fits the 24GB smoke GPU and complies better.
    generator = LocalStructuredGenerator(
        "Qwen/Qwen2.5-7B-Instruct", settings.device
    )

    log("building hierarchy...")
    documents = documents_from_paper_records(papers)
    hierarchy = HierarchyBuilder(settings).build(documents)
    log(f"children={len(hierarchy.child_ids)}")

    base_metadata = {
        "task": "faithful smoke",
        "seed": args.seed,
        "papers": [str(p["paper_id"]) for p in papers],
        "question_ids": [str(q["question_id"]) for q in questions],
        "manifest_hashes": {
            "dev_papers": sha256_file(dev_papers_path),
            "dev_questions": sha256_file(dev_questions_path),
            "manifest_metadata": sha256_file(metadata_path),
        },
        "settings": settings.to_dict(),
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "nli": settings.nli_model,
            "generator": "local:Qwen/Qwen2.5-7B-Instruct (API quotas exhausted 2026-09-19)",
            "longrag_reader": "local:Qwen/Qwen2.5-7B-Instruct (API quotas exhausted 2026-09-19)",
        },
        **git_info(),
    }

    if not args.skip_raptor:
        run_dir = PROJECT_ROOT / "artifacts" / "baselines" / "raptor" / "smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        errors: list[dict] = []
        try:
            log("building B5_raptor_faithful pipeline (SBERT + LLM summaries)...")
            pipeline = make_baseline_pipeline(
                "B5_raptor_faithful", hierarchy, index_factory=index_factory,
                reranker=reranker, generator=generator, verifier=verifier,
                settings=settings,
            )
            (run_dir / "index_metadata.json").write_text(
                json.dumps(pipeline.retriever.index_metadata(), indent=1),
                encoding="utf-8",
            )
            run = run_benchmark(
                "B5_raptor_faithful",
                _RetryAnswerWrapper(pipeline), questions, seed=args.seed,
            )
            traces = []
            for row in run.rows:
                try:
                    traces.append(
                        pipeline.retriever.retrieval_trace(
                            row["query"], row.get("source")
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append({"where": "raptor_trace", "error": str(exc)})
            write_jsonl(run.rows, run_dir / "per_query_predictions.jsonl")
            write_jsonl(traces, run_dir / "retrieval_traces.jsonl")
            (run_dir / "summary_metrics.json").write_text(
                json.dumps(run.summary, indent=1), encoding="utf-8")
            (run_dir / "config.json").write_text(
                json.dumps(settings.to_dict(), indent=1), encoding="utf-8")
            (run_dir / "run_metadata.json").write_text(
                json.dumps({**base_metadata, "baseline": "B5_raptor_faithful",
                            "valid_rows": len(run.rows),
                            "errors": len(errors)}, indent=1), encoding="utf-8")
            log(f"raptor summary: {json.dumps(run.summary)}")
        except Exception:  # noqa: BLE001
            errors.append({"where": "raptor_run", "error": traceback.format_exc()})
            log(f"RAPTOR SMOKE FAILED:\n{traceback.format_exc()}")
        write_jsonl(errors, run_dir / "errors.jsonl")

    if not args.skip_longrag:
        run_dir = PROJECT_ROOT / "artifacts" / "baselines" / "longrag" / "smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        errors = []
        try:
            log("building B6_longrag_faithful pipeline (bge-large-en-v1.5 units)...")
            pipeline = make_baseline_pipeline(
                "B6_longrag_faithful", hierarchy, index_factory=index_factory,
                reranker=reranker, generator=generator, verifier=verifier,
                settings=settings,
            )
            (run_dir / "index_metadata.json").write_text(
                json.dumps(pipeline.retriever.index_metadata(), indent=1),
                encoding="utf-8",
            )
            run = run_benchmark(
                "B6_longrag_faithful_controlled",
                _RetryAnswerWrapper(pipeline), questions, seed=args.seed,
            )
            traces = []
            for row in run.rows:
                try:
                    traces.append(
                        pipeline.retriever.retrieval_trace(
                            row["query"], row.get("source")
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append({"where": "longrag_trace", "error": str(exc)})
            write_jsonl(run.rows, run_dir / "per_query_predictions.jsonl")
            write_jsonl(traces, run_dir / "retrieval_traces.jsonl")
            (run_dir / "summary_metrics.json").write_text(
                json.dumps(run.summary, indent=1), encoding="utf-8")
            # Primary long-context reader pass (free-form answers).
            reader_rows: list[dict] = []
            try:
                from edahr.baselines_longrag import (  # noqa: E402
                    LocalFreeReader,
                    build_document_units,
                )

                # Local reader: API quotas exhausted; substitution recorded.
                reader = LocalFreeReader(
                    model="Qwen/Qwen2.5-7B-Instruct", device=settings.device
                )
                units = {
                    unit.unit_id: unit
                    for unit in build_document_units(hierarchy)
                }
                for record in questions:
                    trace = pipeline.retriever.retrieval_trace(
                        record["query"], record.get("source")
                    )
                    picked = [
                        units[item["unit_id"]] for item in trace["picked"]
                        if item["unit_id"] in units
                    ]
                    answer = reader.generate_answer(record["query"], picked)
                    references = [str(a) for a in record.get("reference_answers") or [""]]
                    predicted_paragraphs = [
                        paragraph_text
                        for unit in picked
                        for _, _, _, paragraph_text in unit.paragraphs
                    ]
                    reader_rows.append({
                        "question_id": record.get("question_id"),
                        "answer": answer,
                        "answer_f1": qasper_answer_token_f1(answer or "Unanswerable", references),
                        "evidence_f1": qasper_evidence_f1(
                            predicted_paragraphs,
                            record.get("reference_evidence_sets") or [],
                        ),
                        "picked_units": [u.unit_id for u in picked],
                    })
            except Exception:  # noqa: BLE001
                errors.append({"where": "longrag_reader", "error": traceback.format_exc()})
                log(f"LONGRAG READER PASS FAILED:\n{traceback.format_exc()}")
            write_jsonl(reader_rows, run_dir / "reader_predictions.jsonl")
            (run_dir / "config.json").write_text(
                json.dumps(settings.to_dict(), indent=1), encoding="utf-8")
            (run_dir / "run_metadata.json").write_text(
                json.dumps({**base_metadata, "baseline": "B6_longrag_faithful",
                            "valid_rows": len(run.rows),
                            "reader_rows": len(reader_rows),
                            "errors": len(errors)}, indent=1), encoding="utf-8")
            log(f"longrag summary: {json.dumps(run.summary)}")
        except Exception:  # noqa: BLE001
            errors.append({"where": "longrag_run", "error": traceback.format_exc()})
            log(f"LONGRAG SMOKE FAILED:\n{traceback.format_exc()}")
        write_jsonl(errors, run_dir / "errors.jsonl")

    for name in ("raptor", "longrag"):
        run_dir = PROJECT_ROOT / "artifacts" / "baselines" / name / "smoke"
        if run_dir.is_dir():
            (run_dir / "execution.log").write_text(
                "\n".join(log_lines) + "\n", encoding="utf-8")
    log("smoke done")


if __name__ == "__main__":
    main()
