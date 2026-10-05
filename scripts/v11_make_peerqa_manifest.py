"""Freeze the PeerQA manifest for v11 (amendment in analysis/v11_protocol.md).

Source: Hugging Face ``mteb/PeerQA`` (CC BY-NC-SA 4.0), the MTEB packaging of
PeerQA's redistributable NLPeer papers with author evidence. Files are read
from data/peerqa/mteb/*.parquet; their SHA-256 and the manifests' SHA-256 are
recorded before any run. Manifests are derived data and are not committed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from edahr.baselines import auto_label_gold_children  # noqa: E402
from edahr.config import Settings  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.peerqa import convert_peerqa, mteb_to_peerqa_rows  # noqa: E402
from edahr.qasper import documents_from_paper_records  # noqa: E402

SOURCE = ROOT / 'data/peerqa/mteb'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import pandas as pd
    frames = {name: pd.read_parquet(SOURCE / f'{name}.parquet') for name in ('corpus', 'queries', 'qrels')}
    rows, qa = mteb_to_peerqa_rows(*(frames[n].to_dict('records') for n in ('corpus', 'queries', 'qrels')))
    papers, questions, report = convert_peerqa(rows, qa)
    settings = replace(Settings.from_json(ROOT / 'artifacts/baselines/main/config.json'), chunk_context='none')
    hierarchy = HierarchyBuilder(settings).build(documents_from_paper_records(papers))
    evaluable = sum(bool(auto_label_gold_children(hierarchy, q)[0]) for q in questions)
    out = ROOT / 'manifests'
    hashes = {}
    for name, items in (('peerqa_papers.jsonl', papers), ('peerqa_questions.jsonl', questions)):
        path = out / name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items),
                        encoding='utf-8', newline='\n')
        hashes[name] = sha(path)
    meta = {
        'source': 'huggingface.co/datasets/mteb/PeerQA (CC BY-NC-SA 4.0), NLPeer subset of PeerQA',
        'source_sha256': {f'{n}.parquet': sha(SOURCE / f'{n}.parquet') for n in frames},
        'conversion': report, 'questions_with_gold_leaves': evaluable,
        'leaves': len(hierarchy.child_ids), 'sha256': hashes,
        'note': 'evidence units are MTEB corpus units (sentence-like); no free-form answers',
    }
    (out / 'peerqa_metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
