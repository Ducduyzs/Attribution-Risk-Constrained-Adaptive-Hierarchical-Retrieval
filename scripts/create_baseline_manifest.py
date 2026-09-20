"""Create frozen QASPER baseline manifests with full provenance.

Writes (repo-root `manifests/`):
  qasper_baseline_dev_papers.jsonl / qasper_baseline_dev_questions.jsonl
  qasper_baseline_test_papers.jsonl / qasper_baseline_test_questions.jsonl
  qasper_baseline_manifest_metadata.json

Selection is seeded and deterministic: papers are shuffled with the given
seed, then taken in order when they contribute at least one question with
gold evidence. Per paper, questions prefer more gold paragraphs (multi-
evidence first) and are capped to spread coverage. No `eligible[:N]` on
raw file order is used anywhere.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.qasper import convert_qasper  # noqa: E402

SCRIPT_VERSION = "create_baseline_manifest v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def select_split(
    papers: list[dict],
    questions: list[dict],
    target_papers: int,
    max_q_per_paper: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    by_paper: dict[str, list[dict]] = {}
    for question in questions:
        if not question.get("gold_quotes"):
            continue
        by_paper.setdefault(str(question["paper_id"]), []).append(question)
    for items in by_paper.values():
        items.sort(
            key=lambda q: (-len(q.get("gold_paragraph_ids") or []), str(q["question_id"]))
        )
    paper_ids = sorted(by_paper)
    rng = random.Random(seed)
    rng.shuffle(paper_ids)
    chosen_papers: list[dict] = []
    chosen_questions: list[dict] = []
    paper_by_id = {str(p["paper_id"]): p for p in papers}
    for paper_id in paper_ids:
        if len(chosen_papers) >= target_papers:
            break
        if paper_id not in paper_by_id:
            continue
        chosen_papers.append(paper_by_id[paper_id])
        chosen_questions.extend(by_paper[paper_id][:max_q_per_paper])
    chosen_questions.sort(key=lambda q: (str(q["paper_id"]), str(q["question_id"])))
    return chosen_papers, chosen_questions


def write_jsonl(rows: list[dict], path: Path) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-papers", type=int, default=30)
    parser.add_argument("--dev-q-per-paper", type=int, default=3)
    parser.add_argument("--test-papers", type=int, default=60)
    parser.add_argument("--test-q-per-paper", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="manifests")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    qasper_dir = PROJECT_ROOT / "data" / "qasper"
    dev_source = qasper_dir / "qasper-dev-v0.3.json"
    test_source = qasper_dir / "qasper-test-v0.3.json"

    dev_papers, dev_questions, _ = convert_qasper(dev_source, "dev")
    test_papers, test_questions, _ = convert_qasper(test_source, "test")

    sel_dev_papers, sel_dev_questions = select_split(
        dev_papers, dev_questions, args.dev_papers, args.dev_q_per_paper, args.seed
    )
    sel_test_papers, sel_test_questions = select_split(
        test_papers, test_questions, args.test_papers, args.test_q_per_paper,
        args.seed + 1,
    )

    hashes = {
        "dev_papers": write_jsonl(sel_dev_papers, out_dir / "qasper_baseline_dev_papers.jsonl"),
        "dev_questions": write_jsonl(sel_dev_questions, out_dir / "qasper_baseline_dev_questions.jsonl"),
        "test_papers": write_jsonl(sel_test_papers, out_dir / "qasper_baseline_test_papers.jsonl"),
        "test_questions": write_jsonl(sel_test_questions, out_dir / "qasper_baseline_test_questions.jsonl"),
    }
    metadata = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "selection_rules": (
            "seeded shuffle of paper ids; keep papers with >=1 gold-evidence "
            "question; per paper prefer questions with more gold paragraphs, "
            f"capped at dev={args.dev_q_per_paper}/paper, test={args.test_q_per_paper}/paper"
        ),
        "splits": {
            "dev": {
                "dataset": "qasper-v0.3", "split": "dev",
                "source_file": str(dev_source.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256_file(dev_source),
                "paper_count": len(sel_dev_papers),
                "question_count": len(sel_dev_questions),
                "papers_manifest_sha256": hashes["dev_papers"],
                "questions_manifest_sha256": hashes["dev_questions"],
            },
            "test": {
                "dataset": "qasper-v0.3", "split": "test",
                "source_file": str(test_source.relative_to(PROJECT_ROOT)),
                "source_sha256": sha256_file(test_source),
                "paper_count": len(sel_test_papers),
                "question_count": len(sel_test_questions),
                "papers_manifest_sha256": hashes["test_papers"],
                "questions_manifest_sha256": hashes["test_questions"],
            },
        },
        **git_info(),
    }
    metadata_path = out_dir / "qasper_baseline_manifest_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {
            "dev_papers": len(sel_dev_papers), "dev_questions": len(sel_dev_questions),
            "test_papers": len(sel_test_papers), "test_questions": len(sel_test_questions),
            **hashes,
        },
        indent=1,
    ))


if __name__ == "__main__":
    main()
