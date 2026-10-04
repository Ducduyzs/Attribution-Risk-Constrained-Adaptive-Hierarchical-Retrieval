"""H1-H3: can the expansion gate learn a generalizable signal?

Grid over the frozen v7 train/dev rollouts (no API calls):

  target   T0  stored v5 label (end-to-end reward gain, one LLM sample,
               computed with the pre-fix verifier guard)
           T1  (H1) generator-independent: expanding adds at least one gold
               leaf that the query had not retrieved (rescue is possible)
  features F0  the 13 stored gate features
           F3  (H3) query-relative: stored features rank-normalised within the
               question + structural features (leaves already retrieved in
               the node, new leaves the expansion would add, member scores
               relative to the question's best group, section position)
  learner  point  (current) pointwise random forest
           pair   (H2) pairwise logistic ranking on within-question feature
                  differences (RankNet-style); scores items with w.x

Reported: paper-grouped 5-fold CV AUC on train, dev AUC with a paper-cluster
bootstrap 95% CI, and within-question pairwise AUC on dev. Acceptance bar
from the readiness audit: dev AUC > 0.60 with the CI excluding 0.5.

T1 can only be positive when the expansion adds a not-yet-retrieved leaf, so
"does the node have unretrieved leaves" alone scores AUC ~0.93. For T1 the
bar is therefore applied to the *conditional* evaluation (rows whose
expansion adds >= 1 new leaf), and that trivial rule is reported alongside.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.config import Settings  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402

FEATURE_DIM = 13


def load_split(rollouts: str, papers: str, metadata: str):
    meta = json.loads((PROJECT_ROOT / metadata).read_text(encoding="utf-8"))
    fields = Settings.__dataclass_fields__
    settings = replace(Settings(), **{k: v for k, v in meta["settings"].items() if k in fields})
    hierarchy = HierarchyBuilder(settings).build(
        documents_from_paper_records(read_jsonl(PROJECT_ROOT / papers))
    )
    rows = [
        json.loads(line) for line in
        (PROJECT_ROOT / rollouts).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = [row for row in rows if row.get("citation_evaluable") and "gold" in row]
    return hierarchy, rows


def enrich(rows: list[dict], hierarchy) -> None:
    """Attach T1 labels and the H3 feature vector to every rollout row."""
    retrieved_by_q: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        retrieved_by_q[row["question_id"]].update(row["members"])
    sections_by_doc: dict[str, list[str]] = defaultdict(list)
    for node_id, node in hierarchy.nodes.items():
        if node.level.value == "section":
            sections_by_doc[node.document_id].append(node_id)
    for row in rows:
        retrieved = retrieved_by_q[row["question_id"]]
        gold = set(row["gold"])
        parent_leaves = set(hierarchy.node(row["parent_id"]).evidence_child_ids)
        section_leaves = set(hierarchy.node(row["section_id"]).evidence_child_ids)
        new_parent = parent_leaves - retrieved
        new_section = section_leaves - parent_leaves - retrieved
        row["T1_parent"] = int(bool(new_parent & gold))
        row["T1_section"] = int(bool(new_section & gold))
        row["adds_parent"] = bool(new_parent)
        row["adds_section"] = bool(new_section)
        scores = list(row["members"].values())
        section = hierarchy.node(row["section_id"])
        doc_sections = sections_by_doc[section.document_id]
        row["_struct"] = [
            len(row["members"]) / max(1, len(parent_leaves)),
            len(new_parent) / max(1, len(parent_leaves)),
            len(new_section) / max(1, len(section_leaves)),
            min(1.0, len(section_leaves) / 40.0),
            max(scores), sum(scores) / len(scores), min(scores),
            doc_sections.index(row["section_id"]) / max(1, len(doc_sections) - 1),
            len(retrieved & section_leaves) / max(1, len(section_leaves)),
        ]
    by_q: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_q[row["question_id"]].append(row)
    for group in by_q.values():
        best = max(max(r["members"].values()) for r in group)
        raw = np.asarray([r["features"][:FEATURE_DIM] for r in group], dtype=float)
        # Within-question rank in [0,1] for every stored feature.
        ranks = raw.argsort(axis=0).argsort(axis=0) / max(1, len(group) - 1)
        for index, row in enumerate(group):
            row["_F3"] = list(ranks[index]) + row["_struct"] + [
                best - max(row["members"].values()), float(len(group) > 1),
            ]


def matrix(rows, feature_set):
    if feature_set == "F0":
        return np.asarray([r["features"][:FEATURE_DIM] for r in rows], dtype=float)
    return np.asarray([r["_F3"] for r in rows], dtype=float)


def auc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def within_question_auc(scores, labels, questions) -> float:
    correct = total = 0.0
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for score, label, question in zip(scores, labels, questions):
        groups[question].append((score, label))
    for items in groups.values():
        for s_pos, l_pos in items:
            if l_pos != 1:
                continue
            for s_neg, l_neg in items:
                if l_neg == 0:
                    total += 1
                    correct += 1.0 if s_pos > s_neg else 0.5 if s_pos == s_neg else 0.0
    return correct / total if total else float("nan")


def fit_score(learner, x_train, y_train, q_train, x_eval, seed):
    if learner == "point":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=4, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ).fit(x_train, y_train)
        return model.predict_proba(x_eval)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    xs = scaler.transform(x_train)
    diffs, signs = [], []
    groups: dict[str, list[int]] = defaultdict(list)
    for index, question in enumerate(q_train):
        groups[question].append(index)
    for members in groups.values():
        for i in members:
            for j in members:
                if y_train[i] > y_train[j]:
                    diffs.extend([xs[i] - xs[j], xs[j] - xs[i]])
                    signs.extend([1, 0])
    if len(set(signs)) < 2:
        return np.zeros(len(x_eval))
    model = LogisticRegression(C=0.5, max_iter=2000, fit_intercept=False).fit(
        np.asarray(diffs), np.asarray(signs)
    )
    return scaler.transform(x_eval) @ model.coef_[0]


def cluster_ci(scores, labels, papers, seed, iterations=1000):
    by_paper: dict[str, list[int]] = defaultdict(list)
    for index, paper in enumerate(papers):
        by_paper[paper].append(index)
    keys = list(by_paper)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        idx = [i for _ in keys for i in by_paper[keys[rng.randrange(len(keys))]]]
        value = auc([scores[i] for i in idx], [labels[i] for i in idx])
        if value == value:
            values.append(value)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="analysis/gate_experiments.json")
    args = parser.parse_args()

    h_train, train = load_split(
        "data/rollouts/rollouts_v7_train_frozen.jsonl",
        "data/manifests_frozen/qasper_train_papers_frozen.jsonl",
        "data/rollouts/rollouts_v7_train_frozen.metadata.json",
    )
    h_dev, dev = load_split(
        "data/rollouts/rollouts_v7_dev.jsonl",
        "data/manifests_frozen/qasper_dev_papers_frozen.jsonl",
        "data/rollouts/rollouts_v7_dev.metadata.json",
    )
    enrich(train, h_train)
    enrich(dev, h_dev)
    from sklearn.model_selection import GroupKFold

    results = []
    for gate in ("parent", "section"):
        targets = {"T0": f"label_{gate}", "T1": f"T1_{gate}"}
        for target, key in targets.items():
            y_tr = np.asarray([r[key] for r in train])
            y_dev = np.asarray([r[key] for r in dev])
            base = {
                "gate": gate, "target": target,
                "train_pos": int(y_tr.sum()), "train_n": len(y_tr),
                "dev_pos": int(y_dev.sum()), "dev_n": len(y_dev),
            }
            for feature_set in ("F0", "F3"):
                x_tr, x_dev = matrix(train, feature_set), matrix(dev, feature_set)
                q_tr = [r["question_id"] for r in train]
                p_tr = [r["source"] for r in train]
                for learner in ("point", "pair"):
                    cv = []
                    for fit_idx, eval_idx in GroupKFold(n_splits=5).split(x_tr, y_tr, p_tr):
                        scores = fit_score(learner, x_tr[fit_idx], y_tr[fit_idx],
                                           [q_tr[i] for i in fit_idx], x_tr[eval_idx], args.seed)
                        cv.append(auc(list(scores), list(y_tr[eval_idx])))
                    scores = list(fit_score(learner, x_tr, y_tr, q_tr, x_dev, args.seed))
                    dev_auc = auc(scores, list(y_dev))
                    low, high = cluster_ci(scores, list(y_dev), [r["source"] for r in dev], args.seed)
                    conditional = None
                    if target == "T1":
                        flag = f"adds_{gate}"
                        sub_tr = [i for i, r in enumerate(train) if r[flag]]
                        sub_dev = [i for i, r in enumerate(dev) if r[flag]]
                        c_scores = list(fit_score(
                            learner, x_tr[sub_tr], y_tr[sub_tr], [q_tr[i] for i in sub_tr],
                            x_dev[sub_dev], args.seed))
                        c_labels = [int(y_dev[i]) for i in sub_dev]
                        c_low, c_high = cluster_ci(
                            c_scores, c_labels, [dev[i]["source"] for i in sub_dev], args.seed)
                        conditional = {
                            "dev_rows": len(sub_dev), "dev_pos": sum(c_labels),
                            "dev_auc": auc(c_scores, c_labels), "dev_auc_ci95": [c_low, c_high],
                            "trivial_has_new_leaves_auc_all_rows": auc(
                                [float(r[flag]) for r in dev], list(y_dev)),
                        }
                    gate_auc, gate_low = (
                        (conditional["dev_auc"], conditional["dev_auc_ci95"][0])
                        if conditional else (dev_auc, low)
                    )
                    row = {
                        **base, "features": feature_set, "learner": learner,
                        "conditional_on_new_leaves": conditional,
                        "cv_auc_mean": float(np.nanmean(cv)), "cv_auc_sd": float(np.nanstd(cv)),
                        "dev_auc": dev_auc, "dev_auc_ci95": [low, high],
                        "dev_within_question_auc": within_question_auc(
                            scores, list(y_dev), [r["question_id"] for r in dev]),
                        "passes_bar": bool(gate_auc > 0.60 and gate_low > 0.5),
                    }
                    results.append(row)
                    print(f"{gate:7} {target} {feature_set} {learner:5} "
                          f"pos(train/dev)={base['train_pos']}/{base['dev_pos']} "
                          f"cvAUC={row['cv_auc_mean']:.3f}±{row['cv_auc_sd']:.3f} "
                          f"devAUC={dev_auc:.3f} [{low:.3f},{high:.3f}] "
                          f"wqAUC={row['dev_within_question_auc']:.3f}"
                          + (f" | cond devAUC={conditional['dev_auc']:.3f} "
                             f"[{conditional['dev_auc_ci95'][0]:.3f},{conditional['dev_auc_ci95'][1]:.3f}] "
                             f"pos={conditional['dev_pos']}/{conditional['dev_rows']}"
                             if conditional else "")
                          + ("  PASS" if row["passes_bar"] else ""), flush=True)

    # How well does the generator-independent target track the end-to-end one?
    agreement = {}
    for gate in ("parent", "section"):
        branch = "parent" if gate == "parent" else "section"
        gains = [r["branches"][branch]["reward"] - r["branches"]["keep"]["reward"]
                 for r in dev if branch in r["branches"]]
        t1 = [r[f"T1_{gate}"] for r in dev if branch in r["branches"]]
        pos = [g for g, t in zip(gains, t1) if t]
        neg = [g for g, t in zip(gains, t1) if not t]
        agreement[gate] = {
            "mean_reward_gain_T1_pos": float(np.mean(pos)) if pos else None,
            "mean_reward_gain_T1_neg": float(np.mean(neg)) if neg else None,
            "auc_T1_predicts_gain_gt_0": auc(t1, [int(g > 0) for g in gains]),
        }
    print(json.dumps(agreement, indent=2))
    (PROJECT_ROOT / args.output).write_text(
        json.dumps({"results": results, "target_agreement_dev": agreement}, indent=2),
        encoding="utf-8",
    )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
