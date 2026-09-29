"""Independent audit of the local QASPER metrics against AllenAI's official evaluator.

The official script is vendored unmodified in ``third_party/qasper_official``
and pinned by SHA-256. Gold references are parsed by the *official* code from
the *raw* QASPER JSON, so this audits the local converter (``convert_qasper``)
and the local metrics together, not the local code against itself.

Part A — full-split agreement: for every question of each raw split, a fixed
battery of synthetic predictions (each annotator's answer/evidence, empty,
"Unanswerable", partial answers, distractor paragraphs, ...) is scored by both
implementations; every per-question difference above ``--tolerance`` is a
mismatch. The corpus-level ``evaluate()`` means are compared as well.

Part B — artifact re-scoring: stored benchmark rows are re-scored with the
official evaluator in its native prediction format and compared with the
values written at run time. Evidence is rebuilt from ``predicted_paragraph_ids``
when ``predicted_evidence_texts`` is absent (rows written before 2026-09-29);
answer F1 needs ``predicted_answer`` and is skipped for older rows.

Exit status is non-zero on any mismatch, so the audit can gate a benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.evaluation import qasper_answer_token_f1, qasper_evidence_f1  # noqa: E402
from edahr.qasper import convert_qasper  # noqa: E402

OFFICIAL_DIR = PROJECT_ROOT / "third_party" / "qasper_official"
OFFICIAL_URL = (
    "https://github.com/allenai/qasper-led-baseline/blob/"
    "e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py"
)
OFFICIAL_SHA256 = "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"


def load_official():
    path = OFFICIAL_DIR / "evaluator.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OFFICIAL_SHA256:
        raise RuntimeError(f"official evaluator hash mismatch: {digest} != {OFFICIAL_SHA256}")
    spec = importlib.util.spec_from_file_location("qasper_official_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def raw_paragraph_texts(raw: dict) -> dict[str, str]:
    """Local paragraph id -> raw paragraph string, mirroring convert_qasper ids."""
    mapping: dict[str, str] = {}
    for paper_id, paper in raw.items():
        for position, section in enumerate(paper.get("full_text") or ()):
            kept = [value for value in section.get("paragraphs") or () if normalize_ws(value)]
            for index, value in enumerate(kept):
                mapping[f"{paper_id}:section:{position}:paragraph:{index}"] = value
        if str(paper.get("abstract") or "").strip():
            mapping[f"{paper_id}:abstract:paragraph:0"] = paper["abstract"]
    return mapping


def prediction_battery(question_id: str, official_refs: list[dict], paper: dict,
                       distractor_answer: str) -> list[tuple[str, str, list[str]]]:
    """Deterministic (case, answer, raw evidence) predictions for one question.

    Evidence is always the raw text of real paragraphs of the paper, which is
    what a paragraph-citing system emits. Annotation strings are *not* used
    verbatim: ~3.6% of them differ from their paragraph in whitespace, and
    the official exact-string match then scores a correct paragraph as a miss.
    """
    paragraphs = [
        value for section in paper.get("full_text") or ()
        for value in section.get("paragraphs") or () if normalize_ws(value)
    ]
    first = paragraphs[:1]
    as_paragraph = {normalize_ws(value): value for value in reversed(paragraphs)}
    if str(paper.get("abstract") or "").strip():
        as_paragraph.setdefault(normalize_ws(paper["abstract"]), paper["abstract"])

    def to_paragraphs(evidence: list[str]) -> list[str]:
        # Drops figure/table evidence ("FLOAT SELECTED"), which is no paragraph.
        return list(dict.fromkeys(
            as_paragraph[normalize_ws(item)] for item in evidence
            if normalize_ws(item) in as_paragraph
        ))

    cases: list[tuple[str, str, list[str]]] = [
        ("empty", "", []),
        ("unanswerable", "Unanswerable", []),
        ("distractor", distractor_answer, first),
        ("first_paragraph", "Yes", first),
        ("first_two_paragraphs", "No", paragraphs[:2]),
    ]
    union: list[str] = []
    for index, reference in enumerate(official_refs):
        answer = reference["answer"]
        tokens = answer.split()
        evidence = to_paragraphs(reference["evidence"])
        union.extend(item for item in evidence if item not in union)
        cases.append((f"annotator{index}", answer, evidence))
        cases.append((f"annotator{index}_half", " ".join(tokens[: max(1, len(tokens) // 2)]),
                      evidence[:1]))
        cases.append((f"annotator{index}_plus_noise", answer + " additionally the baseline",
                      evidence + [item for item in first if item not in evidence]))
    cases.append(("evidence_union", official_refs[0]["answer"] if official_refs else "", union))
    return cases


def audit_split(official, raw_path: Path, tolerance: float) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    gold = official.get_answers_and_evidence(raw, text_evidence_only=False)
    _, records, _ = convert_qasper(raw_path, raw_path.stem)
    by_id = {record["question_id"]: record for record in records}
    paper_of = {qa["question_id"]: paper_id for paper_id, paper in raw.items()
                for qa in paper.get("qas") or ()}
    missing = sorted(set(gold) - set(by_id))
    extra = sorted(set(by_id) - set(gold))

    question_ids = [qid for qid in gold if qid in by_id]
    mismatches: list[dict] = []
    comparisons = 0
    corpus: dict[str, tuple[dict, list[float], list[float]]] = {}
    for position, qid in enumerate(question_ids):
        record = by_id[qid]
        distractor = gold[question_ids[(position + 1) % len(question_ids)]][0]["answer"]
        for case, answer, evidence in prediction_battery(
            qid, gold[qid], raw[paper_of[qid]], distractor
        ):
            official_answer = max(official.token_f1_score(answer, ref["answer"]) for ref in gold[qid])
            official_evidence = max(
                official.paragraph_f1_score(evidence, ref["evidence"]) for ref in gold[qid]
            )
            # Local pipeline predictions are whitespace-normalised paragraph texts.
            local_answer = qasper_answer_token_f1(answer, record["reference_answers"])
            local_evidence = qasper_evidence_f1(
                [normalize_ws(item) for item in evidence], record["reference_evidence_sets"]
            )
            comparisons += 1
            for metric, local_value, official_value in (
                ("answer_f1", local_answer, official_answer),
                ("evidence_f1", local_evidence, official_evidence),
            ):
                if abs(local_value - official_value) > tolerance:
                    mismatches.append({
                        "question_id": qid, "case": case, "metric": metric,
                        "local": round(local_value, 6), "official": round(official_value, 6),
                        "prediction_answer": answer[:200],
                        "official_references": [ref["answer"][:200] for ref in gold[qid]],
                        "local_references": [ref[:200] for ref in record["reference_answers"]],
                    })
            predictions, local_a, local_e = corpus.setdefault(case, ({}, [], []))
            predictions[qid] = {"answer": answer, "evidence": evidence}
            local_a.append(local_answer)
            local_e.append(local_evidence)

    corpus_rows = {}
    for case, (predictions, local_a, local_e) in corpus.items():
        if len(predictions) != len(question_ids):
            continue  # annotator-k cases only exist for questions with >k annotations
        subset_gold = {qid: gold[qid] for qid in question_ids}
        result = official.evaluate(subset_gold, predictions)
        corpus_rows[case] = {
            "official_answer_f1": result["Answer F1"],
            "local_answer_f1": sum(local_a) / len(local_a),
            "official_evidence_f1": result["Evidence F1"],
            "local_evidence_f1": sum(local_e) / len(local_e),
        }
    by_metric: dict[str, int] = {}
    by_case: dict[str, int] = {}
    for item in mismatches:
        by_metric[item["metric"]] = by_metric.get(item["metric"], 0) + 1
        by_case[item["case"]] = by_case.get(item["case"], 0) + 1
    return {
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "questions": len(question_ids),
        "questions_missing_locally": missing,
        "questions_extra_locally": extra,
        "comparisons": comparisons,
        "mismatches": len(mismatches),
        "mismatches_by_metric": by_metric,
        "mismatches_by_case": by_case,
        "questions_with_mismatch": len({item["question_id"] for item in mismatches}),
        "mismatch_examples": mismatches[:25],
        "corpus_level": corpus_rows,
    }


def audit_artifacts(official, raw_paths: list[Path], artifact_paths: list[Path],
                    tolerance: float) -> list[dict]:
    raw: dict = {}
    for path in raw_paths:
        raw.update(json.loads(path.read_text(encoding="utf-8-sig")))
    gold = official.get_answers_and_evidence(raw, text_evidence_only=False)
    paragraph_text = raw_paragraph_texts(raw)
    normalized_to_raw = {normalize_ws(value): value for value in paragraph_text.values()}
    reports = []
    for path in artifact_paths:
        evidence_mismatch: list[dict] = []
        answer_mismatch: list[dict] = []
        rows = scored_answer = unresolved = skipped = 0
        stored_scores: list[float] = []
        official_scores: list[float] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid not in gold or "official_qasper_evidence_f1" not in row:
                continue
            rows += 1
            if "predicted_evidence_texts" in row:
                evidence = [normalized_to_raw.get(normalize_ws(t), t)
                            for t in row["predicted_evidence_texts"]]
            elif "predicted_paragraph_ids" not in row:
                skipped += 1  # e.g. LongRAG reader rows: only unit ids stored
                continue
            else:
                ids = row.get("predicted_paragraph_ids") or []
                unresolved += sum(pid not in paragraph_text for pid in ids)
                evidence = [paragraph_text[pid] for pid in ids if pid in paragraph_text]
            official_evidence = max(
                official.paragraph_f1_score(evidence, ref["evidence"]) for ref in gold[qid]
            )
            stored_scores.append(float(row["official_qasper_evidence_f1"]))
            official_scores.append(official_evidence)
            if abs(official_evidence - float(row["official_qasper_evidence_f1"])) > tolerance:
                evidence_mismatch.append({
                    "question_id": qid, "stored": row["official_qasper_evidence_f1"],
                    "official": round(official_evidence, 6),
                })
            if "predicted_answer" in row:
                scored_answer += 1
                official_answer = max(
                    official.token_f1_score(row["predicted_answer"], ref["answer"])
                    for ref in gold[qid]
                )
                if abs(official_answer - float(row["answer_f1"])) > tolerance:
                    answer_mismatch.append({
                        "question_id": qid, "stored": row["answer_f1"],
                        "official": round(official_answer, 6),
                    })
        reports.append({
            "artifact": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
            "rows": rows,
            "rows_not_rescorable": skipped,
            "unresolved_paragraph_ids": unresolved,
            "stored_mean_evidence_f1": (
                sum(stored_scores) / len(stored_scores) if stored_scores else None
            ),
            "official_mean_evidence_f1": (
                sum(official_scores) / len(official_scores) if official_scores else None
            ),
            "evidence_f1_mismatches": len(evidence_mismatch),
            "answer_f1_rows_scored": scored_answer,
            "answer_f1_mismatches": len(answer_mismatch),
            "examples": (evidence_mismatch + answer_mismatch)[:10],
        })
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", default=[
        "data/qasper/qasper-dev-v0.3.json", "data/qasper/qasper-test-v0.3.json",
    ])
    parser.add_argument("--artifacts", nargs="*", default=[
        str(path.relative_to(PROJECT_ROOT))
        for path in sorted((PROJECT_ROOT / "artifacts/baselines/main").glob("*_rows.jsonl"))
    ])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", default="analysis/evaluator_audit.json")
    args = parser.parse_args()

    official = load_official()
    raw_paths = [PROJECT_ROOT / path for path in args.raw]
    report = {
        "official_evaluator": {
            "url": OFFICIAL_URL, "sha256": OFFICIAL_SHA256, "license": "Apache-2.0",
            "vendored_path": str((OFFICIAL_DIR / "evaluator.py").relative_to(PROJECT_ROOT)),
            "mode": "text_evidence_only=False (official default)",
        },
        "tolerance": args.tolerance,
        "splits": [audit_split(official, path, args.tolerance) for path in raw_paths],
        "artifacts": audit_artifacts(
            official, raw_paths, [PROJECT_ROOT / path for path in args.artifacts], args.tolerance
        ),
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    failed = False
    for split in report["splits"]:
        print(f"[split] {split['raw_file']}: {split['questions']} questions, "
              f"{split['comparisons']} predictions, {split['mismatches']} mismatches "
              f"{split['mismatches_by_metric']} on {split['questions_with_mismatch']} questions")
        failed |= bool(split["mismatches"] or split["questions_missing_locally"])
    for item in report["artifacts"]:
        means = ""
        if item["official_mean_evidence_f1"] is not None:
            means = (f", evidence F1 stored {item['stored_mean_evidence_f1']:.4f} "
                     f"vs official {item['official_mean_evidence_f1']:.4f}")
        print(f"[artifact] {item['artifact']}: {item['rows']} rows "
              f"({item['rows_not_rescorable']} not re-scorable), evidence mismatches "
              f"{item['evidence_f1_mismatches']}{means}, answer rows "
              f"{item['answer_f1_rows_scored']} (mismatches {item['answer_f1_mismatches']})")
        failed |= bool(item["evidence_f1_mismatches"] or item["answer_f1_mismatches"])
    print(f"report: {output.relative_to(PROJECT_ROOT)}")
    print("AUDIT FAILED" if failed else "AUDIT PASSED: local metrics match the official evaluator")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
