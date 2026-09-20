"""Main faithful-baseline evaluation on the frozen dev manifest (Task 9).

Phases (separate processes recommended for VRAM hygiene):
  faithful : B5_raptor_faithful + B6_longrag_faithful (controlled, shared stack)
  flat     : B3_flat_neural + B4_static_hierarchy + edahr(prior)
  reader   : LongRAG primary long-context reader (local Qwen-7B)
  compare  : paired stats + report from phase outputs (CPU only)

Outputs: artifacts/baselines/main/{config.json,run_metadata.json,
  <system>_rows.jsonl,<system>_summary.json,index_meta_<system>.json,
  errors.jsonl,comparison.json,report.md,execution.log}
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


def gpu_peak_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="faithful",
                        choices=["faithful", "flat", "reader", "compare"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config.local.json")
    parser.add_argument("--max-questions", type=int, default=0)
    args = parser.parse_args()

    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_lines: list[str] = []

    def log(message: str) -> None:
        stamped = f"{datetime.datetime.now().isoformat()} {message}"
        log_lines.append(stamped)
        print(stamped, flush=True)

    from edahr.config import Settings  # noqa: E402

    manifest_dir = PROJECT_ROOT / "manifests"
    out_dir = PROJECT_ROOT / "artifacts" / "baselines" / "main"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_papers_path = manifest_dir / "qasper_baseline_dev_papers.jsonl"
    dev_questions_path = manifest_dir / "qasper_baseline_dev_questions.jsonl"
    metadata_path = manifest_dir / "qasper_baseline_manifest_metadata.json"
    settings = Settings.from_json(PROJECT_ROOT / args.config)
    settings = replace(settings, seed=args.seed)

    papers = load_jsonl(dev_papers_path)
    questions = load_jsonl(dev_questions_path)
    if args.max_questions:
        questions = questions[: args.max_questions]
        paper_ids = {str(q["paper_id"]) for q in questions}
        papers = [p for p in papers if str(p["paper_id"]) in paper_ids]
    log(f"main papers={len(papers)} questions={len(questions)} phase={args.phase}")

    base_metadata = {
        "task": "faithful main evaluation (dev)",
        "phase": args.phase,
        "seed": args.seed,
        "papers": len(papers),
        "questions": len(questions),
        "manifest_hashes": {
            "dev_papers": sha256_file(dev_papers_path),
            "dev_questions": sha256_file(dev_questions_path),
            "manifest_metadata": sha256_file(metadata_path),
        },
        "settings": settings.to_dict(),
        **git_info(),
    }

    if args.phase == "compare":
        from edahr.baselines import (  # noqa: E402
            BenchmarkRun,
            clustered_ci_vs_baseline,
            significance_vs_baseline,
        )

        def load_run(name: str) -> BenchmarkRun:
            rows = load_jsonl(out_dir / f"{name}_rows.jsonl")
            summary = json.loads((out_dir / f"{name}_summary.json").read_text())
            return BenchmarkRun(name=name, rows=rows, summary=summary)

        names = ["B5_raptor_faithful", "B6_longrag_faithful_controlled",
                 "B3_flat_neural", "B4_static_hierarchy", "edahr_prior"]
        runs = {name: load_run(name) for name in names}
        baseline = runs["B3_flat_neural"]
        comparison = {}
        for name, run in runs.items():
            if name == "B3_flat_neural":
                continue
            comparison[name] = {
                "paired_p_vs_B3": significance_vs_baseline(
                    run, baseline, "citation_f1", seed=args.seed),
                "clustered_ci_vs_B3": list(clustered_ci_vs_baseline(
                    run, baseline, "citation_f1", seed=args.seed)),
            }
        (out_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=1), encoding="utf-8")
        lines = ["# Main eval (dev, 30 papers/75 questions) — faithful baselines",
                 "",
                 "| system | recall@5 | citation F1 | answer F1 | official ev F1 | tokens |",
                 "|---|---:|---:|---:|---:|---:|"]
        for name in names:
            summary = runs[name].summary
            lines.append(
                f"| {name} | {summary.get('recall@5', 0):.4f} | "
                f"{summary.get('citation_f1', 0):.4f} | "
                f"{summary.get('answer_f1', 0):.4f} | "
                f"{summary.get('official_qasper_evidence_f1', 0):.4f} | "
                f"{summary.get('context_tokens', 0):.1f} |")
        lines += ["", "## Paired vs B3 (citation F1)", ""]
        for name, stats in comparison.items():
            low, high = stats["clustered_ci_vs_B3"]
            lines.append(
                f"- {name}: p={stats['paired_p_vs_B3']:.4f}, "
                f"clustered 95% CI [{low:.4f}, {high:.4f}]")
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log("compare done")
        (out_dir / "execution_compare.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8")
        return

    from edahr.baselines import make_baseline_pipeline, run_benchmark  # noqa: E402
    from edahr.hierarchy import HierarchyBuilder  # noqa: E402
    from edahr.models import (  # noqa: E402
        BGEReranker,
        LocalStructuredGenerator,
        NliVerifier,
    )
    from edahr.qasper import documents_from_paper_records  # noqa: E402

    documents = documents_from_paper_records(papers)
    hierarchy = HierarchyBuilder(settings).build(documents)
    log(f"children={len(hierarchy.child_ids)}")

    reranker = BGEReranker(settings.reranker_model, settings.device)
    verifier = NliVerifier(settings.nli_model, settings.device)
    generator = LocalStructuredGenerator("Qwen/Qwen2.5-7B-Instruct", settings.device)

    _encoder = None

    def index_factory(variant: Settings):
        nonlocal _encoder
        from edahr.index import MultiRepresentationIndex
        from edahr.models import BGEM3Encoder

        if _encoder is None:
            _encoder = BGEM3Encoder(
                settings.embedding_model, settings.device, settings.use_fp16)
        return MultiRepresentationIndex(hierarchy, _encoder, variant)

    errors: list[dict] = []

    def run_system(name: str, display: str) -> None:
        try:
            log(f"building {name}...")
            pipeline = make_baseline_pipeline(
                name, hierarchy, index_factory=index_factory,
                reranker=reranker, generator=generator, verifier=verifier,
                settings=settings)
            try:
                (out_dir / f"index_meta_{display}.json").write_text(
                    json.dumps(pipeline.retriever.index_metadata(), indent=1),
                    encoding="utf-8")
            except (AttributeError, TypeError):
                pass
            run = run_benchmark(display, pipeline, questions, seed=args.seed)
            write_jsonl(run.rows, out_dir / f"{display}_rows.jsonl")
            (out_dir / f"{display}_summary.json").write_text(
                json.dumps({**run.summary, "gpu_peak_gb": gpu_peak_gb()},
                           indent=1),
                encoding="utf-8")
            log(f"{display}: "
                f"rec5={run.summary.get('recall@5', 0):.4f} "
                f"citF1={run.summary.get('citation_f1', 0):.4f} "
                f"ansF1={run.summary.get('answer_f1', 0):.4f}")
        except Exception:  # noqa: BLE001
            errors.append({"where": display, "error": traceback.format_exc()})
            log(f"{display} FAILED:\n{traceback.format_exc()}")

    if args.phase == "faithful":
        run_system("B5_raptor_faithful", "B5_raptor_faithful")
        run_system("B6_longrag_faithful", "B6_longrag_faithful_controlled")
    elif args.phase == "flat":
        run_system("B3_flat_neural", "B3_flat_neural")
        run_system("B4_static_hierarchy", "B4_static_hierarchy")
        run_system("edahr", "edahr_prior")
    elif args.phase == "reader":
        from edahr.baselines_longrag import (  # noqa: E402
            LocalFreeReader,
            build_document_units,
        )
        from edahr.evaluation import (  # noqa: E402
            qasper_answer_token_f1,
            qasper_evidence_f1,
        )

        try:
            pipeline = make_baseline_pipeline(
                "B6_longrag_faithful", hierarchy, index_factory=index_factory,
                reranker=reranker, generator=generator, verifier=verifier,
                settings=settings)
            import gc

            import torch

            # Free everything except the reader path: reranker/verifier/trees
            # stay on CPU pickles; drop GPU caches before long prompts.
            del reranker, verifier, generator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reader = LocalFreeReader("Qwen/Qwen2.5-7B-Instruct", settings.device)
            units = {u.unit_id: u for u in build_document_units(hierarchy)}
            rows: list[dict] = []
            for record in questions:
                try:
                    trace = pipeline.retriever.retrieval_trace(
                        record["query"], record.get("source"))
                    picked = [units[i["unit_id"]] for i in trace["picked"]
                              if i["unit_id"] in units]
                    answer = reader.generate_answer(record["query"], picked)
                    references = [str(a) for a in record.get("reference_answers") or [""]]
                    rows.append({
                        "question_id": record.get("question_id"),
                        "answer": answer,
                        "answer_f1": qasper_answer_token_f1(
                            answer or "Unanswerable", references),
                        "evidence_f1": qasper_evidence_f1(
                            [t for u in picked for _, _, _, t in u.paragraphs],
                            record.get("reference_evidence_sets") or []),
                        "picked_units": [u.unit_id for u in picked],
                    })
                except Exception:  # noqa: BLE001
                    errors.append({"where": "reader_row",
                                   "error": traceback.format_exc()[:500]})
            write_jsonl(rows, out_dir / "B6_longrag_reader_rows.jsonl")
            log(f"reader rows: {len(rows)}/{len(questions)}")
        except Exception:  # noqa: BLE001
            errors.append({"where": "reader", "error": traceback.format_exc()})
            log(f"reader FAILED:\n{traceback.format_exc()}")

    write_jsonl(errors, out_dir / f"errors_{args.phase}.jsonl")
    (out_dir / "config.json").write_text(
        json.dumps(settings.to_dict(), indent=1), encoding="utf-8")
    (out_dir / f"run_metadata_{args.phase}.json").write_text(
        json.dumps({**base_metadata, "errors": len(errors),
                    "gpu_peak_gb": gpu_peak_gb()}, indent=1), encoding="utf-8")
    (out_dir / f"execution_{args.phase}.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8")
    log("phase done")


if __name__ == "__main__":
    main()
