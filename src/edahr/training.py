"""Train the attribution-risk merge policy from counterfactual rollout rows.

Input: JSONL produced by :mod:`edahr.rollouts`. Each row carries the 10-dim
``MergeFeatures`` vector of a candidate parent and binary labels for the two
level steps (``label_parent``, ``label_section``) derived from counterfactual
rewards. Queries are grouped before splitting so no query leaks across the
train/holdout boundary.

Output: a TorchScript MLP emitting merge logits; :class:`AdaptiveMergePolicy`
loads it via ``checkpoint=`` and applies sigmoid to get P(merge), which then
*replaces* the hand-tuned utility comparison inside ``decide_candidates``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
import random
from pathlib import Path
from typing import Sequence


FEATURE_DIM = 14


def load_rollout_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _labels(rows: Sequence[dict], key: str) -> tuple[list[list[float]], list[int]]:
    features = []
    for row in rows:
        vector = [float(v) for v in row["features"][:FEATURE_DIM]]
        while len(vector) < FEATURE_DIM:
            vector.append(0.0)  # legacy 10-dim rows
        features.append(vector)
    targets = [int(row[key]) for row in rows]
    return features, targets


def group_split(
    rows: Sequence[dict], val_ratio: float = 0.25, seed: int = 42,
    by: str = "query",
) -> tuple[list[dict], list[dict]]:
    """Split rows so no group of `by` (query or paper source) straddles sides."""
    keys = sorted({str(row.get(by) or row["query"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(keys)
    holdout = set(keys[: max(1, round(len(keys) * val_ratio))])
    train = [row for row in rows if str(row.get(by) or row["query"]) not in holdout]
    val = [row for row in rows if str(row.get(by) or row["query"]) in holdout]
    return train, val


def v5_label(row: dict, branch: str, epsilon: float, delta: float, tau: float) -> int:
    """Constrained oracle: EXPAND only when precision is preserved within
    ``epsilon`` of KEEP, harmful drift stays under ``delta``, and the reward
    improves by more than ``tau``."""
    keep = row["branches"]["keep"]
    expand = row["branches"].get(branch)
    if not expand or "v5" not in keep or "v5" not in expand:
        return 0
    feasible = (
        expand["v5"]["citation_precision"] >= keep["v5"]["citation_precision"] - epsilon
        and expand["v5"]["citation_recall"] >= keep["v5"]["citation_recall"] - epsilon
        and expand["v5"]["harmful_rate"] <= delta
        and not (
            bool(expand["v5"].get("empty_evidence"))
            and not bool(keep["v5"].get("empty_evidence"))
        )
    )
    gain = expand["reward"] - keep["reward"]
    return int(feasible and gain > tau)


def relabel_v5(
    rows: Sequence[dict], branch: str, epsilon: float = 0.02,
    delta: float = 0.05, tau: float = 0.02,
) -> list[dict]:
    """Return copies of rows whose label for `branch` follows the constrained rule."""
    key = f"label_{branch}" if branch != "section" else "label_section"
    out = []
    for row in rows:
        copy = dict(row)
        copy[key] = v5_label(row, branch, epsilon, delta, tau)
        out.append(copy)
    return out


def _accuracy(logits: list[float], labels: Sequence[int], threshold: float = 0.5) -> float:
    correct = sum(
        1 for logit, label in zip(logits, labels) if (logit >= threshold) == bool(label)
    )
    return correct / max(1, len(labels))


def _rank_auc(logits: list[float], labels: Sequence[int]) -> float:
    positives = [(logit, 1) for logit, label in zip(logits, labels) if label]
    negatives = [(logit, 0) for logit, label in zip(logits, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = ties = 0
    for pos_logit, _ in positives:
        for neg_logit, _ in negatives:
            if pos_logit > neg_logit:
                wins += 1
            elif pos_logit == neg_logit:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def train_label(
    rows: Sequence[dict],
    label_key: str,
    epochs: int = 400,
    patience: int = 60,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden: int = 16,
    seed: int = 42,
    min_margin: float = 0.0,
) -> tuple[object, dict]:
    """Fit a small MLP on one binary level-step label; returns (model, report).

    ``min_margin`` drops rows whose reward gap between the compared branches is
    ambiguous (|gap| <= min_margin), trading coverage for label cleanliness.
    """
    import torch
    from torch import nn

    gap_keys = {
        "label_parent": ("parent", "keep"),
        "label_section": ("section", "keep"),
    }

    def _gap(row: dict) -> float:
        branch, base = gap_keys[label_key]
        return row["branches"].get(branch, {}).get("reward", 0.0) - \
            row["branches"].get(base, {}).get("reward", 0.0)

    if min_margin > 0.0:
        kept = [row for row in rows if abs(_gap(row)) > min_margin]
    else:
        kept = list(rows)
    torch.manual_seed(seed)
    train_rows, val_rows = group_split(kept, by="source")
    x_train, y_train = _labels(train_rows, label_key)
    x_val, y_val = _labels(val_rows, label_key)
    if not x_train or not x_val:
        raise ValueError(f"not enough labelled rows for {label_key}")

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_v = torch.tensor(x_val, dtype=torch.float32)

    model: nn.Module = nn.Sequential(
        nn.Linear(FEATURE_DIM, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    stale = 0
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_v).flatten().tolist()
        val_loss = loss_fn(torch.tensor(val_logits).unsqueeze(1),
                           torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
        if float(val_loss) < best_val_loss - 1e-5:
            best_val_loss = float(val_loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        train_logits = model(x_t).flatten().tolist()
        val_logits = model(x_v).flatten().tolist()

    positive_rate = sum(y_train) / len(y_train)
    report = {
        "label": label_key,
        "rows_total": len(rows),
        "rows_kept_min_margin": len(kept),
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "epochs_run": epoch + 1,
        "positive_rate_train": round(positive_rate, 4),
        "train_acc": round(_accuracy(train_logits, y_train), 4),
        "val_acc": round(_accuracy(val_logits, y_val), 4),
        "val_auc": round(_rank_auc(val_logits, y_val), 4),
        "always_merge_acc": round(positive_rate, 4),
        "never_merge_acc": round(1.0 - positive_rate, 4),
    }
    return model, report


def export_checkpoint(model: object, out_path: str | Path) -> Path:
    import torch

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(out))
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint_metadata(
    checkpoint: str | Path,
    report: dict,
    *,
    source_rollouts: str | Path,
    seed: int,
    min_margin: float,
    v5: bool,
    epsilon: float,
    delta: float,
    tau: float,
    out_path: str | Path | None = None,
) -> Path:
    """Write a sidecar manifest sufficient to identify a trained gate."""
    checkpoint_path = Path(checkpoint).resolve()
    rollout_path = Path(source_rollouts).resolve()
    output = (
        Path(out_path)
        if out_path is not None
        else checkpoint_path.with_suffix(".metadata.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "feature_dim": FEATURE_DIM,
        "source_rollouts": str(rollout_path),
        "source_rollouts_sha256": sha256_file(rollout_path),
        "seed": seed,
        "min_margin": min_margin,
        "v5_constraints": {"enabled": v5, "epsilon": epsilon, "delta": delta, "tau": tau},
        "training_report": report,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
