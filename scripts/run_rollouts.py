"""Generate schema-rich counterfactual rollouts from paper-scoped QA JSONL.

Each input row needs ``query`` and ``source`` (PDF filename). Optional fields:
``question_id``, ``answer``/``gold_answer``, and evidence ``gold_quotes``.
Gold leaf IDs are derived after hierarchy construction when not supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.baselines import auto_label_gold_children  # noqa: E402
from edahr.config import Settings  # noqa: E402
from edahr.rollouts import RolloutRunner  # noqa: E402
from edahr.runtime import build_pipeline  # noqa: E402


def load_records(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("query") or not record.get("source"):
                raise ValueError(f"row {index} requires query and source")
            record["question_id"] = str(record.get("question_id") or f"q{index:06d}")
            record["source"] = Path(str(record["source"])).name
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="question JSONL")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=4)
    args = parser.parse_args()

    records_path = Path(args.records)
    if not records_path.is_absolute():
        records_path = PROJECT_ROOT / records_path
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = PROJECT_ROOT / pdf_dir
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    records = load_records(records_path, args.limit)
    pdf_paths = sorted({pdf_dir / str(record["source"]) for record in records})
    missing = [str(path) for path in pdf_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing referenced PDFs:\n" + "\n".join(missing))

    settings = Settings.from_json(config_path)
    pipeline = build_pipeline(pdf_paths, settings)
    enriched: list[dict] = []
    for record in records:
        copy = dict(record)
        if not copy.get("gold_child_ids"):
            child_ids, _ = auto_label_gold_children(pipeline.hierarchy, copy)
            copy["gold_child_ids"] = sorted(child_ids)
        copy["citation_evaluable"] = bool(copy.get("gold_child_ids"))
        enriched.append(copy)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    rows = RolloutRunner(
        pipeline, settings=settings, samples=args.samples,
        max_groups_per_query=args.max_groups,
    ).run_records(enriched, out)
    evaluable = sum(bool(row.get("citation_evaluable")) for row in rows)
    print(json.dumps({
        "output": str(out), "rows": len(rows),
        "citation_evaluable_rows": evaluable,
        "citation_evaluable_rate": evaluable / max(1, len(rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
