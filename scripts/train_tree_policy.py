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


def positive_probabilities(model, matrix) -> list[float]:
    probabilities = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        return [0.0] * len(matrix)
    return probabilities[:, classes.index(1)].tolist()


def group_cv_selection(
    rows: list[dict], labels: list[int], estimator_names: list[str], seed: int,
    selection_metric: str,
) -> tuple[str, list[dict]]:
    """Select model family on train papers only with group-disjoint folds."""
    from sklearn.model_selection import GroupKFold

    groups = [str(row.get("source") or row.get("question_id") or index)
              for index, row in enumerate(rows)]
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return estimator_names[0], [{
            "estimator": name, "folds": 0, "selection_score": None,
            "status": "insufficient_distinct_papers",
        } for name in estimator_names]
    folds = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=folds)
    matrix = features(rows)
    reports: list[dict] = []
    for name in estimator_names:
        fold_scores: list[float] = []
        for train_index, validation_index in splitter.split(matrix, labels, groups):
            train_rows = [rows[index] for index in train_index]
            validation_rows = [rows[index] for index in validation_index]
            model = estimator(name, seed)
            model.fit(
                matrix[train_index], [labels[index] for index in train_index],
                sample_weight=paper_sample_weights(train_rows),
            )
            probabilities = positive_probabilities(model, matrix[validation_index])
            fold_scores.append(metrics(
                probabilities, [labels[index] for index in validation_index], 0.5,
                paper_sample_weights(validation_rows),
            )[selection_metric])
        reports.append({
            "estimator": name, "folds": folds,
            "fold_scores": fold_scores,
            "selection_score": round(sum(fold_scores) / len(fold_scores), 4),
        })
    selected = max(reports, key=lambda report: report["selection_score"])
    return str(selected["estimator"]), reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True, choices=("parent", "section"))
    parser.add_argument("--estimator", choices=("auto", "rf", "gb", "hgb"), default="auto")
    parser.add_argument("--selection-metric", choices=("accuracy", "balanced_accuracy"),
                        default="balanced_accuracy")
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
    estimator_names = ["rf", "gb", "hgb"] if args.estimator == "auto" else [args.estimator]
    selected_estimator, candidate_models = group_cv_selection(
        train_rows, y_train, estimator_names, args.seed, args.selection_metric
    )
    model = estimator(selected_estimator, args.seed)
    model.fit(features(train_rows), y_train, sample_weight=train_weights)
    p_train = positive_probabilities(model, features(train_rows))
    p_dev = positive_probabilities(model, features(dev_rows))

    candidates = [index / 100 for index in range(5, 96)]
    threshold_table = [
        {"threshold": threshold, **metrics(p_dev, y_dev, threshold, dev_weights)}
        for threshold in candidates
    ]
    selected_threshold = max(
        threshold_table,
        key=lambda row: (row[args.selection_metric], row["balanced_accuracy"], row["accuracy"]),
    )
    threshold = selected_threshold["threshold"]
    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({
        "schema_version": 2, "model": model, "threshold": threshold,
        "feature_dim": FEATURE_DIM, "label": args.label,
    }, out)
    report = {
        "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "estimator": selected_estimator,
        "estimator_request": args.estimator, "threshold": threshold,
        "selection_metric": args.selection_metric,
        "candidate_models": candidate_models,
        "candidate_thresholds": threshold_table,
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
