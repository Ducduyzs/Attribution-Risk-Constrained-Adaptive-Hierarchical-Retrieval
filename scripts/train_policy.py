"""CLI: train the attribution-risk merge policy from rollout JSONL.

Usage:
    python scripts/train_policy.py --rollouts data/rollouts_t0v5_train.jsonl \
        --out checkpoints/policy_v5.ts --label label_parent --v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.training import (  # noqa: E402
    export_checkpoint,
    load_rollout_rows,
    relabel_v5,
    train_label,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--out", default="checkpoints/policy_v0.ts")
    parser.add_argument("--label", default="label_parent",
                        choices=["label_parent", "label_section"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="drop rows with |reward gap| <= margin (label noise filter)")
    parser.add_argument("--v5", action="store_true",
                        help="constrained attribution-risk labels from stored V/R/G sets")
    parser.add_argument("--epsilon", type=float, default=0.02,
                        help="allowed precision drop vs KEEP in the v5 constraint")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="harmful-drift ceiling in the v5 constraint")
    args = parser.parse_args()

    rows = load_rollout_rows(PROJECT_ROOT / args.rollouts)
    if args.v5:
        rows = relabel_v5(rows, args.label.removeprefix("label_"),
                          epsilon=args.epsilon, delta=args.delta)
    model, report = train_label(rows, args.label, seed=args.seed, min_margin=args.min_margin)
    out = export_checkpoint(model, PROJECT_ROOT / args.out)
    report["checkpoint"] = str(out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
