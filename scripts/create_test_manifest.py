"""Create stratified test manifest for publication-grade evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.qasper import read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_answer_type(question: dict) -> str:
    """Derive a reproducible answer category from all available annotations."""
    query = str(question.get("query") or "").strip().lower()
    references = [
        str(value).strip() for value in question.get("reference_answers") or ()
        if str(value).strip()
    ]
    answerable = [value for value in references if value.lower() != "unanswerable"]
    if question.get("all_annotations_unanswerable") or not answerable:
        return "unanswerable"
    if all(value.lower() in {"yes", "no", "true", "false"} for value in answerable):
        return "boolean"
    if re.search(r"\b(how many|how much|number of|count of)\b", query):
        return "numeric"
    if re.search(r"\b(list|enumerate|name|what are|which are)\b", query):
        return "list"
    lengths = sorted(len(value.split()) for value in answerable)
    median_length = lengths[len(lengths) // 2]
    return "short" if median_length <= 5 else "long"


def classify_query_type(query: str) -> str:
    """Classify query type."""
    query_lower = query.lower()
    if any(word in query_lower for word in ["compare", "contrast", "difference", "versus", " vs "]):
        return "comparative"
    if query_lower.startswith(("what", "which", "who", "when", "where")):
        return "factoid"
    elif query_lower.startswith(("how", "why", "explain", "describe")):
        return "explanatory"
    else:
        return "global"


def paper_word_count(paper: dict) -> int:
    """Count words from the actual frozen-paper schema."""
    return sum(
        len(str(section.get("text") or "").split())
        for section in paper.get("sections") or ()
    )


def get_paper_length_bucket(paper: dict, thresholds: tuple[int, int]) -> str:
    """Bucket paper by empirical word-count tertiles."""
    total_words = paper_word_count(paper)
    if total_words <= thresholds[0]:
        return "short"
    elif total_words <= thresholds[1]:
        return "medium"
    else:
        return "long"


def get_evidence_bucket(question: dict) -> str:
    """Bucket question by number of evidence paragraphs."""
    evidence_count = max(
        (len(values) for values in question.get("reference_paragraph_sets") or ()),
        default=0,
    )
    if evidence_count <= 2:
        return "few"
    elif evidence_count <= 5:
        return "medium"
    else:
        return "many"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="data/manifests_frozen")
    parser.add_argument("--paper-manifest", default="qasper_test_papers_frozen.jsonl")
    parser.add_argument("--question-manifest", default="qasper_test_questions_frozen.jsonl")
    parser.add_argument("--out-dir", default="data/manifests_test")
    parser.add_argument("--target-papers", type=int, default=180)
    parser.add_argument("--max-q-per-paper", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = PROJECT_ROOT / manifest_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = list(read_jsonl(manifest_dir / args.paper_manifest))
    questions = list(read_jsonl(manifest_dir / args.question_manifest))

    # Filter to citation-evaluable only
    questions = [q for q in questions if q.get("gold_quotes")]

    # Build paper info
    paper_info = {}
    for p in papers:
        paper_info[p["paper_id"]] = p

    eligible_paper_ids = {
        q["paper_id"] for q in questions if q["paper_id"] in paper_info
    }
    word_counts = sorted(
        paper_word_count(paper_info[paper_id]) for paper_id in eligible_paper_ids
    )
    if not word_counts:
        raise ValueError("no citation-evaluable test papers")
    length_thresholds = (
        word_counts[(len(word_counts) - 1) // 3],
        word_counts[(2 * (len(word_counts) - 1)) // 3],
    )

    rng = random.Random(args.seed)
    questions_by_paper = defaultdict(list)
    for question in questions:
        if question["paper_id"] in paper_info:
            questions_by_paper[question["paper_id"]].append(question)
    representatives = []
    for paper_id in sorted(questions_by_paper):
        candidates = sorted(
            questions_by_paper[paper_id],
            key=lambda row: str(row.get("question_id") or ""),
        )
        representatives.append(candidates[rng.randrange(len(candidates))])

    # Stratify questions
    strata = defaultdict(list)
    for q in representatives:
        paper = paper_info.get(q["paper_id"])
        if not paper:
            continue
        answer_type = classify_answer_type(q)
        query_type = classify_query_type(q["query"])
        length_bucket = get_paper_length_bucket(paper, length_thresholds)
        evidence_bucket = get_evidence_bucket(q)
        stratum_key = f"{answer_type}|{query_type}|{length_bucket}|{evidence_bucket}"
        strata[stratum_key].append(q)

    print(f"Total strata: {len(strata)}")
    for key, items in sorted(strata.items()):
        print(f"  {key}: {len(items)} questions")

    # Allocate an exact paper target using largest remainders.
    total_questions = sum(len(items) for items in strata.values())
    target_questions = min(args.target_papers, total_questions)
    exact = {
        key: target_questions * len(items) / total_questions
        for key, items in strata.items()
    }
    allocation = {
        key: min(len(strata[key]), math.floor(value))
        for key, value in exact.items()
    }
    remaining = target_questions - sum(allocation.values())
    order = sorted(
        strata,
        key=lambda key: (
            exact[key] - math.floor(exact[key]), len(strata[key]), key
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate requested unique papers")

    final_questions = []
    for key in sorted(strata):
        items = list(strata[key])
        rng.shuffle(items)
        final_questions.extend(items[:allocation[key]])
    final_questions.sort(
        key=lambda row: (
            str(row["paper_id"]), str(row.get("question_id") or "")
        )
    )
    paper_counts = {q["paper_id"]: 1 for q in final_questions}

    final_papers = [p for p in papers if p["paper_id"] in paper_counts]

    print(f"\nSelected: {len(final_papers)} papers, {len(final_questions)} questions")
    print(f"Avg questions/paper: {len(final_questions) / max(1, len(final_papers)):.2f}")

    # Write manifests
    q_path = out_dir / "qasper_test_stratified_questions.jsonl"
    p_path = out_dir / "qasper_test_stratified_papers.jsonl"

    with q_path.open("w", encoding="utf-8") as f:
        for q in final_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with p_path.open("w", encoding="utf-8") as f:
        for p in final_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Create selection metadata
    metadata = {
        "schema_version": 1,
        "dataset": "qasper-v0.3",
        "split": "test-stratified",
        "seed": args.seed,
        "target_papers": args.target_papers,
        "max_questions_per_paper": 1,
        "paper_count": len(final_papers),
        "question_count": len(final_questions),
        "paper_length_unit": "whitespace_words",
        "paper_length_tertile_thresholds": list(length_thresholds),
        "population_strata_distribution": {
            k: len(v) for k, v in strata.items()
        },
        "selected_strata_distribution": {
            k: allocation[k] for k in sorted(allocation) if allocation[k]
        },
        "selection_sha256": sha256(q_path),
        "paper_manifest_sha256": sha256(manifest_dir / args.paper_manifest),
        "question_manifest_sha256": sha256(manifest_dir / args.question_manifest),
    }

    meta_path = out_dir / "test_manifest_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nManifests written to {out_dir}")
    print(f"Selection SHA256: {metadata['selection_sha256']}")


if __name__ == "__main__":
    main()
