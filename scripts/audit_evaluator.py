"""Audit evaluator against official QASPER evaluator on regression fixture."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.evaluation import (
    qasper_answer_exact_match,
    qasper_answer_token_f1,
    qasper_evidence_f1,
    citation_precision,
    citation_recall,
    citation_f1,
    evidence_span_recall,
)


# Multi-annotator regression fixture based on QASPER paper examples
REGRESSION_FIXTURE = [
    {
        "question_id": "test_001",
        "prediction": "The model uses BERT for encoding.",
        "references": ["The model uses BERT for encoding.", "BERT is used as the encoder."],
        "predicted_evidence": ["p1", "p2"],
        "reference_evidence_sets": [["p1", "p3"], ["p2", "p4"]],
        "gold_quotes": ["The model uses BERT for encoding.", "BERT serves as the encoder."],
        "predicted_quotes": ["The model uses BERT for encoding.", "BERT is used for encoding."],
    },
    {
        "question_id": "test_002",
        "prediction": "Unanswerable",
        "references": ["Unanswerable", "The paper does not mention this."],
        "predicted_evidence": [],
        "reference_evidence_sets": [[]],
        "gold_quotes": [],
        "predicted_quotes": [],
    },
    {
        "question_id": "test_003",
        "prediction": "They compare to BERT, RoBERTa, and XLNet.",
        "references": ["The baselines include BERT, RoBERTa, and XLNet.", "Comparison is against BERT, RoBERTa, XLNet."],
        "predicted_evidence": ["p5", "p6", "p7"],
        "reference_evidence_sets": [["p5", "p6"], ["p5", "p7"]],
        "gold_quotes": ["We compare to BERT, RoBERTa, and XLNet.", "Baselines: BERT, RoBERTa, XLNet."],
        "predicted_quotes": ["They compare to BERT, RoBERTa, and XLNet.", "BERT, RoBERTa, and XLNet are baselines."],
    },
    {
        "question_id": "test_004",
        "prediction": "The dataset contains 5000 examples.",
        "references": ["There are 5000 examples in the dataset.", "The dataset size is 5000."],
        "predicted_evidence": ["p8"],
        "reference_evidence_sets": [["p8", "p9"], ["p8"]],
        "gold_quotes": ["The dataset contains 5000 examples.", "5000 examples are in the dataset."],
        "predicted_quotes": ["The dataset contains 5000 examples."],
    },
    {
        "question_id": "test_005",
        "prediction": "Training takes 3 days on 8 GPUs.",
        "references": ["Training requires 3 days on 8 GPUs.", "3 days on 8 GPUs for training."],
        "predicted_evidence": ["p10"],
        "reference_evidence_sets": [["p10", "p11"]],
        "gold_quotes": ["Training takes 3 days on 8 GPUs.", "3 days on 8 GPUs."],
        "predicted_quotes": ["Training takes 3 days on 8 GPUs.", "3 days on 8 GPUs for training."],
    },
]


def run_audit():
    print("=" * 60)
    print("EVALUATOR AUDIT: Custom vs Expected (Manual)")
    print("=" * 60)

    for item in REGRESSION_FIXTURE:
        qid = item["question_id"]
        print(f"\n--- {qid} ---")

        # Answer EM
        answer_em = qasper_answer_exact_match(item["prediction"], item["references"])
        print(f"  Answer EM: {answer_em:.4f}")

        # Answer F1
        answer_f1 = qasper_answer_token_f1(item["prediction"], item["references"])
        print(f"  Answer F1: {answer_f1:.4f}")

        # Evidence F1 (official multi-annotator)
        evidence_f1 = qasper_evidence_f1(item["predicted_evidence"], item["reference_evidence_sets"])
        print(f"  Official Evidence F1: {evidence_f1:.4f}")

        # Citation metrics (leaf-level)
        pred_evidence = set(item["predicted_evidence"])
        # For citation metrics, we need gold leaf evidence
        # Here we simulate by using the first reference set as gold
        gold_evidence = set(item["reference_evidence_sets"][0]) if item["reference_evidence_sets"] else set()
        cite_p = citation_precision(pred_evidence, gold_evidence)
        cite_r = citation_recall(pred_evidence, gold_evidence)
        cite_f1 = citation_f1(pred_evidence, gold_evidence)
        print(f"  Citation P/R/F1 (vs first ref): {cite_p:.4f} / {cite_r:.4f} / {cite_f1:.4f}")

        # Evidence span recall
        span_recall = evidence_span_recall(item["predicted_quotes"], item["gold_quotes"])
        print(f"  Evidence Span Recall: {span_recall:.4f}")

    # Test edge cases
    print("\n" + "=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)

    # Empty prediction
    print("\nEmpty prediction:")
    print(f"  EM: {qasper_answer_exact_match('', ['answer']):.4f}")
    print(f"  F1: {qasper_answer_token_f1('', ['answer']):.4f}")
    print(f"  Evidence F1: {qasper_evidence_f1([], [[]]):.4f}")

    # Empty reference
    print("\nEmpty reference:")
    print(f"  EM: {qasper_answer_exact_match('answer', ['']):.4f}")
    print(f"  F1: {qasper_answer_token_f1('answer', ['']):.4f}")

    # Multiple references - max over annotators
    print("\nMultiple references (max over annotators):")
    print(f"  EM: {qasper_answer_exact_match('cat', ['dog', 'cat']):.4f}")
    print(f"  F1: {qasper_answer_token_f1('cat', ['dog', 'cat']):.4f}")

    # Evidence F1 with multiple reference sets
    print(f"  Evidence F1: {qasper_evidence_f1(['p1'], [['p1'], ['p2', 'p3']]):.4f}")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_audit()