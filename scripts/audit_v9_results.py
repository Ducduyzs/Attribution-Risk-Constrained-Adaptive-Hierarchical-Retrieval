"""Validate complete v9 matrix, budgets, provenance, and official DEV scoring."""
import json
from collections import Counter
from pathlib import Path

from audit_evaluator import audit_artifacts, load_official

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / 'artifacts/v9_results'
rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((base / 'rows').glob('*.json'))]
expected_systems = ['flat24', 'all_leaf', 'contextual24', 'contextual_all', 'mmr', 'coverage',
                    'sentence', 'window', 'dependency', 'evidence_first', 'dependency_first', 'raptor']
counts = Counter((r['system'], r['budget']) for r in rows)
errors = []
for system in expected_systems:
    for budget in (512, 1024, 2048):
        if counts[(system, budget)] != 75:
            errors.append(f'{system}@{budget}: {counts[(system, budget)]}/75')
for r in rows:
    if r['context_tokens'] > r['budget']:
        errors.append('budget violation ' + r['question_id'])
    if r['generation']['validation_errors']:
        errors.append('generation contract ' + r['question_id'])
    blocks = {b['context_id']: b for b in r['context']}
    for e in r['evidence']:
        # Quote may be a union of cited fragments from the same leaf.
        raw_claim = next((c for c in r['generation']['claims'] if c['text'] == e['claim_text']), None)
        fragments = [blocks[cid]['text'] for cid in raw_claim['citations'] if cid in blocks and blocks[cid]['node_id'] == e['node_id']] if raw_claim else []
        if not fragments or any(part not in fragments for part in e['quote'].split('\n')):
            errors.append('hidden verifier evidence ' + r['question_id'])
official = audit_artifacts(load_official(), [ROOT / 'data/qasper/qasper-dev-v0.3.json'],
                          sorted(base.glob('artifacts_*.jsonl')), 1e-9)
for item in official:
    if item['evidence_f1_mismatches'] or item['answer_f1_mismatches']:
        errors.append('official evaluator mismatch ' + item['artifact'])
unique_calls = {}
for path in (base / 'cache/generator').glob('*.json'):
    data = json.loads(path.read_text(encoding='utf-8'))
    unique_calls[path.stem] = data['usage']
report = {'complete': not errors, 'rows': len(rows), 'systems': len(expected_systems),
    'budgets': [512, 1024, 2048], 'errors': errors,
    'official_artifact_audits': official,
    'unique_generation_calls': len(unique_calls),
    'unique_api_prompt_tokens': sum(u['prompt_tokens'] for u in unique_calls.values()),
    'unique_api_completion_tokens': sum(u['completion_tokens'] for u in unique_calls.values()),
    'answer_word_limit_violations': sum(r['over_answer_word_limit'] for r in rows),
    'test_accessed': False,
    'human_attribution_audit': 'unlabelled; requires independent human assessment'}
(ROOT / 'analysis/v9_execution_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'official_artifact_audits'}, indent=2))
raise SystemExit(0 if not errors else 1)
