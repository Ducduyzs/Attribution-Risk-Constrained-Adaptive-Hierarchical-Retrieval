"""Why does RAPTOR pack more gold into a budget than flat reranking? (offline, CPU)

Uses the cached v9 rankings (no GPU, no API). Gold labels are used only to
MEASURE packed contexts, never to select them. Everything here is DEV and
exploratory: any selector designed from it must be confirmed on unseen papers.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import run_v9_directions as v9  # noqa: E402
from edahr.baselines import auto_label_gold_children  # noqa: E402


def setup():
    from dataclasses import replace
    import tiktoken
    from edahr.config import Settings
    from edahr.hierarchy import HierarchyBuilder
    from edahr.qasper import documents_from_paper_records

    class Args:
        questions = 75
    papers, records, _ = v9.load_dev(Args)
    settings = replace(Settings.from_json(ROOT / 'artifacts/baselines/main/config.json'),
                       chunk_context='none')
    hierarchy = HierarchyBuilder(settings).build(documents_from_paper_records(papers))
    prepared = [json.loads((ROOT / 'artifacts/v9_results/prepared' /
                            (v9.digest(r['question_id']) + '.json')).read_text())
                for r in records]
    encoder = tiktoken.get_encoding('cl100k_base')
    return records, prepared, hierarchy, settings, (lambda text: len(encoder.encode(text)))


def main() -> None:
    records, prepared, hierarchy, settings, count = setup()
    stats = {k: [] for k in ('pool_all', 'pool_raptor', 'gold_rank_all', 'gold_rank_raptor',
                             'above_gold_all', 'above_gold_removed_by_raptor',
                             'raptor_has_gold', 'packed_all', 'packed_raptor',
                             'gold_in_top3_all', 'gold_in_top3_raptor')}
    for record, data in zip(records, prepared):
        gold, _ = auto_label_gold_children(hierarchy, record)
        if not gold:
            continue
        all_ids = [cid for cid, _ in data['rankings']['all_leaf']]
        rap_ids = [cid for cid, _ in data['rankings']['raptor']]
        best_all = min((i for i, c in enumerate(all_ids) if c in gold), default=None)
        best_rap = min((i for i, c in enumerate(rap_ids) if c in gold), default=None)
        stats['pool_all'].append(len(all_ids))
        stats['pool_raptor'].append(len(rap_ids))
        stats['raptor_has_gold'].append(best_rap is not None)
        if best_all is not None:
            stats['gold_rank_all'].append(best_all + 1)
            above = all_ids[:best_all]
            stats['above_gold_all'].append(len(above))
            stats['above_gold_removed_by_raptor'].append(sum(c not in rap_ids for c in above))
            stats['gold_in_top3_all'].append(best_all < 3)
        if best_rap is not None:
            stats['gold_rank_raptor'].append(best_rap + 1)
        stats['gold_in_top3_raptor'].append(best_rap is not None and best_rap < 3)
        for arm, key in (('all_leaf', 'packed_all'), ('raptor', 'packed_raptor')):
            context, _, _ = v9.make_context(arm, data, hierarchy, settings, 1024, count)
            stats[key].append(len(context))
    mean = lambda xs: st.mean([float(x) for x in xs]) if xs else float('nan')
    report = {k: round(mean(v), 3) for k, v in stats.items()}
    report['questions_with_gold'] = len(stats['pool_all'])
    print(json.dumps(report, indent=2))
    (ROOT / 'analysis/v10_packing_analysis.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
