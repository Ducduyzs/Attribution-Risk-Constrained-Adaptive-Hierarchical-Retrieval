"""Freeze protocol: create manifests, lock dependencies, seeds."""

import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    manifest_dir = PROJECT_ROOT / 'data' / 'manifests'
    out_dir = PROJECT_ROOT / 'data' / 'manifests_frozen'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frozen test papers (40 papers from v5)
    frozen_test_papers = {
        '1603.07252', '1604.02038', '1606.07043', '1611.04234', '1611.06322',
        '1703.10152', '1708.00549', '1708.05521', '1805.04833', '1805.07882',
        '1805.11937', '1806.04387', '1808.04122', '1810.02229', '1810.02268',
        '1812.00382', '1903.02930', '1904.09131', '1905.00472', '1905.00840',
        '1906.08871', '1907.04072', '1907.05338', '1907.10676', '1908.11049',
        '1909.00091', '1909.00437', '1909.04387', '1909.09070', '1909.10481',
        '1909.12079', '1911.04474', '1911.04873', '1911.10742', '1912.02866',
        '2003.07568', '2003.11687', '2003.13016', '2004.01820', '2004.04124'
    }

    # Diagnostic papers from v6 runs
    diagnostic_papers = set()
    artifacts_dir = PROJECT_ROOT / 'data' / 'artifacts'
    for d in artifacts_dir.glob('v6*'):
        if d.is_dir():
            for f in d.glob('artifacts_*.jsonl'):
                rows = [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines()]
                for r in rows:
                    diagnostic_papers.add(r.get('source', '').replace('.qasper', ''))

    excluded_papers = frozen_test_papers | diagnostic_papers
    print(f'Excluded papers: {len(excluded_papers)} total')
    print(f'  Frozen test: {len(frozen_test_papers)}')
    print(f'  Diagnostic: {len(diagnostic_papers)}')
    print(f'  Overlap: {len(frozen_test_papers & diagnostic_papers)}')

    SEED = 42
    MAX_Q_PER_PAPER = 3

    for split in ['train', 'dev', 'test']:
        papers = [json.loads(l) for l in (manifest_dir / f'qasper_{split}_papers.jsonl').read_text(encoding='utf-8').splitlines()]
        questions = [json.loads(l) for l in (manifest_dir / f'qasper_{split}_questions.jsonl').read_text(encoding='utf-8').splitlines()]

        # Filter to citation-evaluable only
        questions = [q for q in questions if q.get('gold_quotes')]

        # Exclude papers
        questions = [q for q in questions if q['paper_id'] not in excluded_papers]
        papers = [p for p in papers if p['paper_id'] not in excluded_papers]

        # Shuffle with fixed seed for reproducibility
        rng = random.Random(SEED)
        rng.shuffle(questions)

        # Select up to MAX_Q_PER_PAPER per paper
        selected = []
        paper_counts = defaultdict(int)
        for q in questions:
            if paper_counts[q['paper_id']] < MAX_Q_PER_PAPER:
                selected.append(q)
                paper_counts[q['paper_id']] += 1

        # Limit total questions for train/dev
        if split == 'train':
            selected = selected[:800]  # 400-800 target
        elif split == 'dev':
            selected = selected[:250]  # 150-250 target
        # test: use all remaining

        # Write selected questions
        sel_path = out_dir / f'qasper_{split}_questions_frozen.jsonl'
        with sel_path.open('w', encoding='utf-8') as f:
            for q in selected:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

        # Write corresponding papers
        used_paper_ids = set(q['paper_id'] for q in selected)
        selected_papers = [p for p in papers if p['paper_id'] in used_paper_ids]
        paper_path = out_dir / f'qasper_{split}_papers_frozen.jsonl'
        with paper_path.open('w', encoding='utf-8') as f:
            for p in selected_papers:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')

        print(f'{split}: {len(selected_papers)} papers, {len(selected)} questions')
        print(f'  citation_evaluable: {len(selected)}')
        print(f'  max q/paper: {max(paper_counts.values()) if paper_counts else 0}')

    # Create protocol manifest
    protocol = {
        'schema_version': 1,
        'created_by': 'freeze_protocol_v7',
        'seed': SEED,
        'max_questions_per_paper': MAX_Q_PER_PAPER,
        'excluded_papers': {
            'frozen_test_v5': sorted(frozen_test_papers),
            'diagnostic_v6': sorted(diagnostic_papers),
            'total_excluded': len(excluded_papers)
        },
        'splits': {
            'train': {'target_range': '400-800', 'paper_disjoint_from_dev_test': True},
            'dev': {'target_range': '150-250', 'paper_disjoint_from_train_test': True},
            'test': {'target_range': '150-200 papers', 'paper_disjoint_from_train_dev': True, 'never_used_for_tuning': True}
        },
        'dependencies': {
            'python': '3.10+',
            'torch': '2.4+',
            'transformers': '4.51+',
            'FlagEmbedding': '1.3.5+',
            'faiss-cpu': '1.10+',
            'docling': '2.48+',
            'openai': '1.40+',
            'google-genai': '1.30+'
        },
        'model_snapshots': {
            'embedding': 'BAAI/bge-m3',
            'reranker': 'BAAI/bge-reranker-v2-m3',
            'nli': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
            'generator_openai': 'gpt-4o-mini-2024-07-18',
            'generator_gemini': 'gemini-1.5-flash-002'
        },
        'prompt_version': 'v6_generation_contract_fixed',
        'verifier_config': {
            'nli_support_threshold': 0.25,
            'nli_contradiction_threshold': 0.50,
            'claim_confidence_threshold': 0.55,
            'lexical_support_min_coverage': 0.8,
            'sibling_threshold_delta': 0.10,
            'evidence_margin': 0.05,
            'max_evidence_per_claim': 1
        }
    }

    protocol_path = out_dir / 'protocol_manifest.json'
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(f'\nProtocol manifest written to {protocol_path}')

    # Verify paper disjointness
    train_papers = set(json.loads(l)['paper_id'] for l in (out_dir / 'qasper_train_questions_frozen.jsonl').read_text(encoding='utf-8').splitlines())
    dev_papers = set(json.loads(l)['paper_id'] for l in (out_dir / 'qasper_dev_questions_frozen.jsonl').read_text(encoding='utf-8').splitlines())
    test_papers = set(json.loads(l)['paper_id'] for l in (out_dir / 'qasper_test_questions_frozen.jsonl').read_text(encoding='utf-8').splitlines())

    print('\nPaper disjointness check:')
    print(f'  train intersect dev: {len(train_papers & dev_papers)} (should be 0)')
    print(f'  train intersect test: {len(train_papers & test_papers)} (should be 0)')
    print(f'  dev intersect test: {len(dev_papers & test_papers)} (should be 0)')

if __name__ == '__main__':
    main()