"""Re-derive QASPER reference fields of existing question manifests in place.

Only ``answer``, ``reference_answers`` and ``reference_evidence_sets`` are
rewritten, from the raw QASPER JSON through the current ``convert_qasper``
(which matches AllenAI's official evaluator; see scripts/audit_evaluator.py).
Question selection, order and every other field are preserved. The script
refuses to write if any other converter-derived field would differ, or if the
file does not round-trip byte-identically through its JSON-lines format.

Metadata files that pin a manifest's SHA-256 are updated, and the previous
hashes are kept under ``reference_refresh`` for provenance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.qasper import convert_qasper  # noqa: E402

REFRESHED = ("answer", "reference_answers", "reference_evidence_sets")
MUST_MATCH = ("paper_id", "source", "query", "reference_paragraph_sets",
              "gold_paragraph_ids", "gold_quotes")
RAW = {
    "train": "data/qasper/qasper-train-v0.3.json",
    "dev": "data/qasper/qasper-dev-v0.3.json",
    "test": "data/qasper/qasper-test-v0.3.json",
}
DEFAULT_MANIFESTS = (
    "data/manifests/qasper_train_questions.jsonl",
    "data/manifests/qasper_dev_questions.jsonl",
    "data/manifests/qasper_test_questions.jsonl",
    "data/manifests_frozen/qasper_train_questions.jsonl",
    "data/manifests_frozen/qasper_train_questions_frozen.jsonl",
    "data/manifests_frozen/qasper_dev_questions.jsonl",
    "data/manifests_frozen/qasper_dev_questions_frozen.jsonl",
    "data/manifests_frozen/qasper_test_questions.jsonl",
    "data/manifests_frozen/qasper_test_questions_frozen.jsonl",
    "data/manifests_test/qasper_test_stratified_questions.jsonl",
    "manifests/qasper_baseline_dev_questions.jsonl",
    "manifests/qasper_baseline_test_questions.jsonl",
)
# metadata file -> JSON key paths holding the SHA-256 of a question manifest.
METADATA = {
    "data/manifests_test/test_manifest_metadata.json": {
        ("selection_sha256",): "data/manifests_test/qasper_test_stratified_questions.jsonl",
        ("question_manifest_sha256",): "data/manifests_frozen/qasper_test_questions_frozen.jsonl",
    },
    "manifests/qasper_baseline_manifest_metadata.json": {
        ("splits", "dev", "questions_manifest_sha256"): "manifests/qasper_baseline_dev_questions.jsonl",
        ("splits", "test", "questions_manifest_sha256"): "manifests/qasper_baseline_test_questions.jsonl",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def refresh(path: Path, fresh: dict[str, dict], write: bool) -> dict:
    original = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in original.splitlines() if line.strip()]
    if "".join(dump(row) for row in rows) != original:
        raise RuntimeError(f"{path}: not in canonical JSON-lines form; refusing to rewrite")
    changed = 0
    for row in rows:
        reference = fresh.get(str(row["question_id"]))
        if reference is None:
            raise KeyError(f"{path}: question {row['question_id']} not in raw QASPER")
        for key in MUST_MATCH:
            if key in row and row[key] != reference[key]:
                raise RuntimeError(f"{path}: {row['question_id']} field {key!r} would change")
        before = [row.get(key) for key in REFRESHED]
        for key in REFRESHED:
            row[key] = reference[key]
        changed += before != [row[key] for key in REFRESHED]
    previous = sha256(path)
    if write:
        # Keep the file's own line endings: pinned hashes are byte hashes.
        newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
        path.write_text("".join(dump(row) for row in rows), encoding="utf-8", newline=newline)
    return {"questions": len(rows), "changed": changed,
            "previous_sha256": previous, "sha256": sha256(path) if write else None}


def get_path(data: dict, keys: tuple[str, ...]):
    for key in keys[:-1]:
        data = data[key]
    return data, keys[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="*", default=DEFAULT_MANIFESTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fresh: dict[str, dict] = {}
    for split, raw in RAW.items():
        _, records, _ = convert_qasper(PROJECT_ROOT / raw, split)
        fresh.update({record["question_id"]: record for record in records})

    results = {}
    for name in args.manifests:
        path = PROJECT_ROOT / name
        if not path.exists():
            print(f"[skip] {name}: missing")
            continue
        results[name] = refresh(path, fresh, write=not args.dry_run)
        print(f"[{'dry' if args.dry_run else 'refreshed'}] {name}: {results[name]}")
    if args.dry_run:
        return

    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    for meta_name, pins in METADATA.items():
        meta_path = PROJECT_ROOT / meta_name
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log = {"refreshed_utc": stamp, "fields": list(REFRESHED),
               "reason": "reference strings/evidence aligned with the official QASPER evaluator",
               "script": "scripts/refresh_manifest_references.py", "previous": {}}
        for keys, manifest in pins.items():
            if manifest not in results:
                continue
            holder, key = get_path(meta, keys)
            if holder.get(key) != results[manifest]["previous_sha256"]:
                raise RuntimeError(
                    f"{meta_name}: {'.'.join(keys)} does not pin {manifest}; not updating"
                )
            log["previous"][".".join(keys)] = holder[key]
            holder[key] = results[manifest]["sha256"]
        if log["previous"]:
            meta.setdefault("reference_refresh", []).append(log)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"[metadata] {meta_name}: {log['previous']}")


if __name__ == "__main__":
    main()
