"""Train a nonlinear v5 gate on train and calibrate its threshold on dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.training import FEATURE_DIM, load_rollout_rows, v5_label  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_with_gold(path: Path) -> list[dict]:
    return [
        row for row in load_rollout_rows(path)
        if row.get("citation_evaluable") and "v5" in row.get("branches", {}).get("keep", {})
    ]


def features(rows: list[dict]):
    import numpy as np

    return np.asarray([
        (list(row["features"][:FEATURE_DIM]) + [0.0] * FEATURE_DIM)[:FEATURE_DIM]
        for row in rows
    ], dtype="float32")


def paper_sample_weights(rows: list[dict]) -> list[float]:
    papers = [str(row.get("source") or row.get("question_id") or index)
              for index, row in enumerate(rows)]
    counts = Counter(papers)
    raw = [1.0 / counts[paper] for paper in papers]
    scale = len(raw) / sum(raw) if raw else 1.0
    return [weight * scale for weight in raw]


def estimator(name: str, seed: int):
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=4,
            class_weight="balanced", random_state=seed,
        )
    if name == "gb":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=2, min_samples_leaf=4,
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(name)


def weighted_auc(probabilities: list[float], labels: list[int], weights: list[float]) -> float:
    positives = [(probability, weight) for probability, label, weight
                 in zip(probabilities, labels, weights) if label]
    negatives = [(probability, weight) for probability, label, weight
                 in zip(probabilities, labels, weights) if not label]
    denominator = sum(weight for _, weight in positives) * sum(
        weight for _, weight in negatives
    )
    if denominator == 0.0:
        return 0.5
    score = 0.0
    for positive, positive_weight in positives:
        for negative, negative_weight in negatives:
            if positive > negative:
                score += positive_weight * negative_weight
            elif positive == negative:
                score += 0.5 * positive_weight * negative_weight
    return score / denominator


def metrics(probabilities: list[float], labels: list[int], threshold: float,
            weights: list[float] | None = None) -> dict:
    weights = weights or [1.0] * len(labels)
    predicted = [value >= threshold for value in probabilities]
    accuracy = sum(
        weight for value, label, weight in zip(predicted, labels, weights)
        if value == bool(label)
    ) / sum(weights)
    positives = [i for i, label in enumerate(labels) if label]
    negatives = [i for i, label in enumerate(labels) if not label]
    positive_weight = sum(weights[i] for i in positives)
    negative_weight = sum(weights[i] for i in negatives)
    tpr = sum(weights[i] for i in positives if predicted[i]) / (positive_weight or 1.0)
    tnr = sum(weights[i] for i in negatives if not predicted[i]) / (negative_weight or 1.0)
    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round((tpr + tnr) / 2, 4),
        "auc": round(weighted_auc(probabilities, labels, weights), 4),
        "positive_rate": round(
            sum(weight for label, weight in zip(labels, weights) if label) / sum(weights), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True, choices=("parent", "section"))
    parser.add_argument("--estimator", choices=("rf", "gb", "hgb"), required=True)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path, dev_path = Path(args.train), Path(args.dev)
    out = Path(args.out)
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    if not dev_path.is_absolute():
        dev_path = PROJECT_ROOT / dev_path
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    train_rows, dev_rows = rows_with_gold(train_path), rows_with_gold(dev_path)
    y_train = [v5_label(row, args.label, args.epsilon, args.delta, args.tau) for row in train_rows]
    y_dev = [v5_label(row, args.label, args.epsilon, args.delta, args.tau) for row in dev_rows]
    train_weights = paper_sample_weights(train_rows)
    dev_weights = paper_sample_weights(dev_rows)
    model = estimator(args.estimator, args.seed)
    model.fit(features(train_rows), y_train, sample_weight=train_weights)
    p_train = model.predict_proba(features(train_rows))[:, 1].tolist()
    p_dev = model.predict_proba(features(dev_rows))[:, 1].tolist()

    candidates = [index / 100 for index in range(5, 96)]
    ranked = [
        (metrics(p_dev, y_dev, threshold, dev_weights)["accuracy"],
         metrics(p_dev, y_dev, threshold, dev_weights)["balanced_accuracy"], threshold)
        for threshold in candidates
    ]
    _, _, threshold = max(ranked, key=lambda item: (item[0], item[1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({
        "schema_version": 1, "model": model, "threshold": threshold,
        "feature_dim": FEATURE_DIM, "label": args.label,
    }, out)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "estimator": args.estimator, "threshold": threshold,
        "seed": args.seed, "feature_dim": FEATURE_DIM,
        "paper_weighting": "inverse_rows_per_source_normalized_mean_one",
        "constraints": {"epsilon": args.epsilon, "delta": args.delta, "tau": args.tau},
        "train_rows_total": len(load_rollout_rows(train_path)),
        "train_rows_citation_evaluable": len(train_rows),
        "dev_rows_total": len(load_rollout_rows(dev_path)),
        "dev_rows_citation_evaluable": len(dev_rows),
        "train_metrics": metrics(p_train, y_train, threshold, train_weights),
        "dev_metrics": metrics(p_dev, y_dev, threshold, dev_weights),
        "train_rollouts": str(train_path.resolve()), "train_sha256": digest(train_path),
        "dev_rollouts": str(dev_path.resolve()), "dev_sha256": digest(dev_path),
        "checkpoint": str(out.resolve()),
    }
    report["checkpoint_sha256"] = digest(out)
    metadata = out.with_suffix(".metadata.json")
    metadata.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
