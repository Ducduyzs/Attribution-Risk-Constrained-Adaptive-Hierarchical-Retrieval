"""Offline headroom/ranking diagnostics for v9 (DEV only, no model/API)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from edahr.baselines import auto_label_gold_children
from edahr.config import Settings
from edahr.evaluation import recall_at_k, reciprocal_rank
from edahr.hierarchy import HierarchyBuilder
from edahr.qasper import documents_from_paper_records, read_jsonl
from edahr.statistics import compare, holm

parser = argparse.ArgumentParser()
parser.add_argument('--prepared', default='artifacts/v9_directions/prepared')
args = parser.parse_args()
papers = read_jsonl(ROOT / 'manifests/qasper_baseline_dev_papers.jsonl')
records = read_jsonl(ROOT / 'manifests/qasper_baseline_dev_questions.jsonl')
config = Settings.from_json(ROOT / 'artifacts/baselines/main/config.json')
h = HierarchyBuilder(config).build(documents_from_paper_records(papers))
lookup = {r['question_id']: r for r in records}
rows = []
for path in sorted((ROOT / args.prepared).glob('*.json')):
    data = json.loads(path.read_text(encoding='utf-8'))
    record = lookup[data['question_id']]
    gold, _ = auto_label_gold_children(h, record)
    for name, ranking in data['rankings'].items():
        ids = [cid for cid, _ in ranking]
        rows.append({'system': name, 'question_id': data['question_id'], 'source': data['source'],
                     'recall5': recall_at_k(ids, gold, 5), 'recall10': recall_at_k(ids, gold, 10),
                     'pool_recall': recall_at_k(ids, gold, len(ids)),
                     'mrr': reciprocal_rank(ids, gold), 'candidates': len(ids)})
names = ['flat24', 'all_leaf', 'contextual24', 'contextual_all', 'raptor']
summaries = {}
for name in names:
    subset = [r for r in rows if r['system'] == name]
    summaries[name] = {k: sum(r[k] for r in subset) / len(subset)
                       for k in ('recall5', 'recall10', 'pool_recall', 'mrr', 'candidates')}
comparisons = []
clusters = {r['question_id']: r['source'] for r in records}
for name in names[1:]:
    for metric in ('recall5', 'recall10', 'pool_recall'):
        result = compare({r['question_id']: r[metric] for r in rows if r['system'] == name},
                         {r['question_id']: r[metric] for r in rows if r['system'] == 'flat24'}, clusters)
        comparisons.append({'system': name, 'metric': metric, **result})
for result, adjusted in zip(comparisons, holm([r['p'] for r in comparisons])):
    result['p_holm'] = adjusted
out = ROOT / 'analysis/v9_ranking_diagnostics.json'
out.write_text(json.dumps({'summaries': summaries, 'comparisons_vs_flat': comparisons, 'rows': rows}, indent=2), encoding='utf-8')
lines = ['# V9 ranking diagnostics (DEV, no new generation)', '',
         '| System | Recall@5 | Recall@10 | Pool recall | MRR | Rerank candidates |',
         '|---|---:|---:|---:|---:|---:|']
for name, row in summaries.items():
    lines.append('| ' + name + ' | ' + ' | '.join(f'{row[k]:.4f}' for k in ('recall5', 'recall10', 'pool_recall', 'mrr', 'candidates')) + ' |')
lines += ['', 'Full paired paper-clustered results and Holm adjustments are in v9_ranking_diagnostics.json.',
          'Gold is used only for scoring. Pool recall is not evidence visible in the packed context.']
(ROOT / 'analysis/v9_ranking_diagnostics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
