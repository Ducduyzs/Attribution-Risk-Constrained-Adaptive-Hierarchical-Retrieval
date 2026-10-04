"""Freeze the v10 confirmation manifest (preregistered in analysis/v10_selector_design.md).

QASPER dev papers never used before: not in the 30 baseline-dev papers and not
in the 164 frozen-dev papers of the v7 gate. One question with gold evidence
per paper, seed 20261004, at most 100 papers. Writes manifests + SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004
MAX_PAPERS = 100


def read(path):
    return [json.loads(l) for l in (ROOT / path).read_text(encoding='utf-8').splitlines() if l.strip()]


def main() -> None:
    papers = read('data/manifests/qasper_dev_papers.jsonl')
    questions = read('data/manifests/qasper_dev_questions.jsonl')
    used = {p['paper_id'] for p in read('manifests/qasper_baseline_dev_papers.jsonl')}
    used |= {p['paper_id'] for p in read('data/manifests_frozen/qasper_dev_papers_frozen.jsonl')}
    by_paper: dict[str, list[dict]] = {}
    for q in questions:
        if q['paper_id'] not in used and q.get('gold_quotes'):
            by_paper.setdefault(q['paper_id'], []).append(q)
    rng = random.Random(SEED)
    chosen_papers = sorted(by_paper)
    rng.shuffle(chosen_papers)
    chosen_papers = sorted(chosen_papers[:MAX_PAPERS])
    chosen_questions = []
    for paper_id in chosen_papers:
        candidates = sorted(by_paper[paper_id], key=lambda q: q['question_id'])
        chosen_questions.append(candidates[rng.randrange(len(candidates))])
    paper_rows = [p for p in papers if p['paper_id'] in set(chosen_papers)]
    out = ROOT / 'manifests'
    files = {
        'qasper_v10_confirm_papers.jsonl': paper_rows,
        'qasper_v10_confirm_questions.jsonl': chosen_questions,
    }
    hashes = {}
    for name, rows in files.items():
        path = out / name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                        encoding='utf-8', newline='\n')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        'purpose': 'v10 agreement-selector confirmation (preregistered)',
        'source': 'data/manifests/qasper_dev_*.jsonl (QASPER v0.3 dev)',
        'excluded': 'baseline-dev 30 papers + frozen-dev 164 papers',
        'seed': SEED, 'papers': len(paper_rows), 'questions': len(chosen_questions),
        'eligible_papers': len(by_paper), 'sha256': hashes, 'test_accessed': False,
    }
    (out / 'qasper_v10_confirm_metadata.json').write_text(json.dumps(meta, indent=2) + '\n',
                                                          encoding='utf-8', newline='\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
