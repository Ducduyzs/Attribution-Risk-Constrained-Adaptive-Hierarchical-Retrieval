"""Offline search for an agreement-based context selector (CPU, no API).

Mechanism found in v10_packing_analysis: RAPTOR's pool removes leaves that the
cross-encoder ranks above the gold but an independent semantic signal does
not support, so the shared packer receives fewer high-scoring distractors.
This script tests whether cheap independent signals already cached by v9 can
play the same role without building RAPTOR trees:

  rerank   cross-encoder on the raw leaf (v9 all_leaf)
  bge      BGE-M3 hybrid first-stage rank (v9 initial_ids)
  sent     best cross-encoder score among the leaf's sentences
  ctx      cross-encoder on the LLM-contextualized leaf (v9 contextual_all)
  raptor   RAPTOR pool membership (upper reference; needs trees)

Fusion = reciprocal-rank fusion (k=60). To keep the shared packer comparable,
the fused ORDER receives the cross-encoder's own score multiset (i-th fused
item gets the i-th highest rerank score), so only the order changes.

Measured with gold (never used to select): packed gold-leaf recall, gold
paragraph character coverage, complete evidence set visible. DEV, exploratory:
the chosen selector must be confirmed on unseen papers before any claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import run_v9_directions as v9  # noqa: E402
from edahr.baselines import auto_label_gold_children  # noqa: E402
from edahr.experimental_v9 import diagnostics  # noqa: E402
from edahr.schemas import Generation  # noqa: E402
from edahr.statistics import compare  # noqa: E402
from v10_packing_analysis import setup  # noqa: E402

K = 60
SIGNAL_SETS = {
    'rerank': ('rerank',),
    'rerank+bge': ('rerank', 'bge'),
    'rerank+sent': ('rerank', 'sent'),
    'rerank+ctx': ('rerank', 'ctx'),
    'rerank+bge+sent': ('rerank', 'bge', 'sent'),
    'rerank+bge+ctx': ('rerank', 'bge', 'ctx'),
    'rerank+sent+ctx': ('rerank', 'sent', 'ctx'),
    'rerank+bge+sent+ctx': ('rerank', 'bge', 'sent', 'ctx'),
    'rerank+raptor (ref)': ('rerank', 'raptor'),
    # Hard agreement filters: keep the top KEEP share of leaves by the second
    # signal(s) (RAPTOR's pool keeps 12.8 / 29.3 = 44% of a paper), then rank
    # the survivors by the cross-encoder.
    'filter:bge': ('filter', 'bge'),
    'filter:ctx': ('filter', 'ctx'),
    'filter:sent': ('filter', 'sent'),
    'filter:bge+ctx': ('filter', 'bge', 'ctx'),
    'filter:bge+sent+ctx': ('filter', 'bge', 'sent', 'ctx'),
    # Hypotheses about what RAPTOR adds (SBERT = RAPTOR's own embedding model):
    'rerank+sbert': ('rerank', 'sbert'),
    'filter:sbert': ('filter', 'sbert'),
    'support': ('support',),
    'rerank+support': ('rerank', 'support'),
    'cluster': ('cluster',),
}
KEEP = 0.44


EMBED: dict = {}


def load_embeddings(records, hierarchy) -> None:
    """SBERT (multi-qa-mpnet-base-cos-v1, RAPTOR's model) leaf/query vectors, cached."""
    import numpy as np
    path = ROOT / 'artifacts/v10/sbert_dev.npz'
    leaf_ids = list(hierarchy.child_ids)
    if path.exists():
        data = np.load(path, allow_pickle=True)
        EMBED.update(leaf=dict(zip(data['leaf_ids'], data['leaf'])),
                     query=dict(zip(data['query_ids'], data['query'])))
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1', device='cpu')
    leaf = model.encode([hierarchy.node(c).text for c in leaf_ids], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=False)
    query = model.encode([r['query'] for r in records], normalize_embeddings=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, leaf_ids=np.array(leaf_ids), leaf=leaf,
             query_ids=np.array([r['question_id'] for r in records]), query=query)
    EMBED.update(leaf=dict(zip(leaf_ids, leaf)), query=dict(zip([r['question_id'] for r in records], query)))


def support_scores(data: dict) -> dict[str, float]:
    """Cross-encoder score plus similarity-weighted support from other leaves."""
    import numpy as np
    ids = [cid for cid, _ in data['rankings']['all_leaf']]
    score = np.array([s for _, s in data['rankings']['all_leaf']])
    vectors = np.stack([EMBED['leaf'][c] for c in ids])
    sim = np.clip(vectors @ vectors.T, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    support = sim @ score / np.maximum(sim.sum(axis=1), 1e-9)
    return dict(zip(ids, score + support))


def cluster_ranking(data: dict) -> list[tuple[str, float]]:
    """RAPTOR without summaries: agglomerative clusters (~4 leaves) of SBERT
    leaf vectors, scored by their best cross-encoder leaf; keep the best
    clusters until KEEP of the paper, ordered by the cross-encoder."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    ranking = data['rankings']['all_leaf']
    ids = [cid for cid, _ in ranking]
    score = dict(ranking)
    if len(ids) < 3:
        return ranking
    labels = AgglomerativeClustering(n_clusters=max(2, round(len(ids) / 4)), metric='cosine',
                                     linkage='average').fit_predict(np.stack([EMBED['leaf'][c] for c in ids]))
    best = {}
    for cid, label in zip(ids, labels):
        best[label] = max(best.get(label, -1e9), score[cid])
    kept, target = [], max(2, round(KEEP * len(ids)))
    for label in sorted(best, key=lambda l: -best[l]):
        kept += [c for c, lab in zip(ids, labels) if lab == label]
        if len(kept) >= target:
            break
    return [(cid, s) for cid, s in ranking if cid in set(kept)]


def signal_orders(data: dict) -> dict[str, list[str]]:
    rerank = [cid for cid, _ in data['rankings']['all_leaf']]
    leaves = set(rerank)
    best_sentence: dict[str, float] = {}
    for unit in data['sentences']:
        best_sentence[unit['leaf_id']] = max(best_sentence.get(unit['leaf_id'], -1e9), unit['score'])
    raptor = [cid for cid, _ in data['rankings']['raptor']]
    return {
        'rerank': rerank,
        'bge': [c for c in data['initial_ids'] if c in leaves] + [c for c in rerank if c not in data['initial_ids']],
        'sent': sorted(rerank, key=lambda c: -best_sentence.get(c, -1e9)),
        'ctx': [cid for cid, _ in data['rankings']['contextual_all']],
        'raptor': raptor + [c for c in rerank if c not in raptor],
        'sbert': sorted(rerank, key=lambda c: -float(EMBED['leaf'][c] @ EMBED['query'][data['question_id']])),
        'support': (lambda sup: sorted(rerank, key=lambda c: -sup[c]))(support_scores(data)),
    }


def fused_ranking(data: dict, signals: tuple[str, ...]) -> list[tuple[str, float]]:
    if signals == ('cluster',):
        return cluster_ranking(data)
    orders = signal_orders(data)
    if signals[0] == 'filter':
        second: dict[str, float] = {}
        for name in signals[1:]:
            for rank, cid in enumerate(orders[name], 1):
                second[cid] = second.get(cid, 0.0) + 1.0 / (K + rank)
        keep = max(2, round(KEEP * len(orders['rerank'])))
        kept = set(sorted(second, key=lambda c: (-second[c], c))[:keep])
        return [(cid, score) for cid, score in data['rankings']['all_leaf'] if cid in kept]
    scores: dict[str, float] = {}
    for name in signals:
        for rank, cid in enumerate(orders[name], 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (K + rank)
    order = sorted(scores, key=lambda c: (-scores[c], c))
    rerank_scores = sorted((s for _, s in data['rankings']['all_leaf']), reverse=True)
    return list(zip(order, rerank_scores))


def main() -> None:
    records, prepared, hierarchy, settings, count = setup()
    golds = [auto_label_gold_children(hierarchy, r)[0] for r in records]
    load_embeddings(records, hierarchy)
    clusters = {r['question_id']: r['source'] for r in records}
    arms = {name: signals for name, signals in SIGNAL_SETS.items()}
    results: dict = {}
    per_question: dict = {}
    for budget in (512, 1024, 2048):
        for arm, signals in [('raptor (v9)', None), *arms.items()]:
            touch, cover, complete = {}, {}, {}
            for record, data, gold in zip(records, prepared, golds):
                if not gold:
                    continue
                if signals is None:
                    context, pool, _ = v9.make_context('raptor', data, hierarchy, settings, budget, count)
                else:
                    variant = dict(data, rankings=dict(data['rankings'], all_leaf=fused_ranking(data, signals)))
                    context, pool, _ = v9.make_context('all_leaf', variant, hierarchy, settings, budget, count)
                d = diagnostics(record, hierarchy, pool, context, Generation(False), {}, gold)
                qid = record['question_id']
                touch[qid] = d['packed_leaf_touch_recall']
                cover[qid] = d['gold_paragraph_char_coverage_best_ref']
                complete[qid] = d['complete_evidence_set_visible']
            per_question[(arm, budget)] = (touch, cover, complete)
            results[f'{arm}@{budget}'] = {
                'packed_gold_leaf_recall': sum(touch.values()) / len(touch),
                'gold_char_coverage': sum(cover.values()) / len(cover),
                'complete_set_visible': sum(complete.values()) / len(complete),
            }
    print(f"{'selector':26} " + ' '.join(f'{b:>22}' for b in (512, 1024, 2048)))
    print(f"{'':26} " + ' '.join(f'{"touch / cover / full":>22}' for _ in range(3)))
    for arm in ['raptor (v9)', *arms]:
        cells = [results[f'{arm}@{b}'] for b in (512, 1024, 2048)]
        print(f'{arm:26} ' + ' '.join(
            f"{c['packed_gold_leaf_recall']:.3f} / {c['gold_char_coverage']:.3f} / {c['complete_set_visible']:.3f}".rjust(22)
            for c in cells))
    print('\nvs raptor (v9), gold char coverage, paper-clustered:')
    for arm in arms:
        cells = []
        for b in (512, 1024, 2048):
            out = compare(per_question[(arm, b)][1], per_question[('raptor (v9)', b)][1], clusters)
            cells.append(f"{out['mean_diff']:+.3f} (p={out['p']:.2f})")
        print(f'{arm:26} ' + '   '.join(cells))
    (ROOT / 'analysis/v10_selector_search.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
