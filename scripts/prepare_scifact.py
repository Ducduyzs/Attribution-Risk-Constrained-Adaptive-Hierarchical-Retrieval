"""Convert the official SciFact release into paper-scoped benchmark manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.qasper import write_jsonl  # noqa: E402
from edahr.scifact import convert_scifact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/scifact/data")
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    papers, questions, report = convert_scifact(
        data_dir / "corpus.jsonl", data_dir / f"claims_{args.split}.jsonl", args.split
    )
    write_jsonl(papers, out_dir / f"scifact_{args.split}_papers.jsonl")
    write_jsonl(questions, out_dir / f"scifact_{args.split}_questions.jsonl")
    (out_dir / f"scifact_{args.split}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()