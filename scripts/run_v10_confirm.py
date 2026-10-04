"""V10 confirmation run (preregistered: analysis/v10_selector_design.md).

Arms (frozen): raptor, all_leaf, agreement. Budgets 512/1024/2048 cl100k tokens.
Manifest: manifests/qasper_v10_confirm_* (100 unused dev papers, hashes checked).
Phases, identical split to v9 so the API key never leaves the local machine:
  prepare  (GPU)   cross-encoder over all leaves, RAPTOR pool, SBERT similarities
  generate (local) gpt-4o-mini answers -> pending/
  verify   (GPU)   visible-only NLI verification -> rows/
  report   (local) preregistered primary analysis
Generation, verification, packing and scoring reuse run_v9_directions unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import run_v9_directions as v9  # noqa: E402
from edahr.agreement import SBERT_MODEL, agreement_ranking  # noqa: E402
from edahr.config import Settings  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402
from edahr.schemas import Claim, Generation  # noqa: E402

SYSTEMS = ['raptor', 'all_leaf', 'agreement']
BUDGETS = [512, 1024, 2048]
VERSION = 'v10-confirm.1'
MARGIN = 0.02
# One-shot test (preregistered in analysis/v10_test_protocol.md). Question hash =
# selection_sha256 of data/manifests_test/test_manifest_metadata.json.
TEST_FILES = {
    'data/manifests_test/qasper_test_stratified_papers.jsonl':
        'a8096fda113f21cc9b59d59b255ae3d3a20569c4a79cc6dad4d20c08b1dfa1ad',
    'data/manifests_test/qasper_test_stratified_questions.jsonl':
        'de0241e23bab67f5c5972bf9af2de1860e885f51c96dc2d7d8e707689c5c2c7b',
}
SPLIT = 'confirm'


def load_manifest():
    if SPLIT == 'test':
        hashes = {}
        for name, expected in TEST_FILES.items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'test manifest hash mismatch: {name}')
            hashes[name] = actual
        papers, records = (read_jsonl(ROOT / name) for name in TEST_FILES)
        return papers, records, hashes
    meta = json.loads((ROOT / 'manifests/qasper_v10_confirm_metadata.json').read_text(encoding='utf-8'))
    paths = {name: ROOT / 'manifests' / name for name in meta['sha256']}
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256'][name]:
            raise RuntimeError(f'manifest hash mismatch: {name}')
    papers = read_jsonl(paths['qasper_v10_confirm_papers.jsonl'])
    records = read_jsonl(paths['qasper_v10_confirm_questions.jsonl'])
    return papers, records, meta['sha256']


def prepare(settings, hierarchy, records, root):
    paths = [root / 'prepared' / (v9.digest(r['question_id']) + '.json') for r in records]
    if all(p.exists() for p in paths):
        return [json.loads(p.read_text()) for p in paths]
    from sentence_transformers import SentenceTransformer
    from edahr.baselines import raptor_faithful_retriever
    from edahr.models import BGEReranker
    reranker = v9.CachedReranker(BGEReranker(settings.reranker_model, settings.device,
                                             settings.use_fp16), root / 'cache/reranker')
    sbert = SentenceTransformer(SBERT_MODEL, device=settings.device)
    print('[prepare] building/loading RAPTOR trees', flush=True)
    t0 = time.perf_counter()
    raptor = raptor_faithful_retriever(hierarchy, settings)
    v9.dump(root / 'raptor_index_seconds.json', {'seconds': time.perf_counter() - t0})
    output = []
    for n, (record, path) in enumerate(zip(records, paths), 1):
        if path.exists():
            output.append(json.loads(path.read_text()))
            continue
        query, source = record['query'], record['source']
        leaves = [cid for cid in hierarchy.child_ids if hierarchy.node(cid).source == source]
        t0 = time.perf_counter()
        raw = dict(zip(leaves, reranker.score(query, [hierarchy.node(c).text for c in leaves])))
        rerank_seconds = time.perf_counter() - t0
        all_leaf = sorted(raw.items(), key=lambda x: -x[1])
        t0 = time.perf_counter()
        vectors = sbert.encode([query] + [hierarchy.node(c).text for c in leaves],
                               normalize_embeddings=True, show_progress_bar=False)
        similarity = {c: float(vectors[0] @ v) for c, v in zip(leaves, vectors[1:])}
        agreement = agreement_ranking(all_leaf, similarity)
        agreement_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        pool = [h.node_id for h in raptor.search(query, settings.candidate_k, source=source)]
        pool = pool[:settings.rerank_k]
        raptor_seconds = time.perf_counter() - t0
        data = {'question_id': record['question_id'], 'source': source, 'query': query,
                'rankings': {'all_leaf': all_leaf, 'agreement': agreement,
                             'raptor': sorted(((c, raw[c]) for c in pool), key=lambda x: -x[1])},
                'similarity': similarity, 'leaf_count': len(leaves),
                'seconds': {'rerank': rerank_seconds, 'sbert': agreement_seconds,
                            'raptor_search': raptor_seconds}}
        v9.dump(path, data)
        output.append(data)
        print(f'[prepare] {n}/{len(records)} leaves={len(leaves)}', flush=True)
    return output


def make_context(system, data, hierarchy, settings, budget, count):
    if system == 'agreement':
        data = dict(data, rankings=dict(data['rankings'], all_leaf=data['rankings']['agreement']))
        return v9.make_context('all_leaf', data, hierarchy, settings, budget, count)
    return v9.make_context(system, data, hierarchy, settings, budget, count)


def report(root, records):
    from edahr.statistics import cluster_bootstrap_ci, cluster_sign_flip_test, holm, non_inferiority
    rows = [json.loads(p.read_text()) for p in sorted((root / 'rows').glob('*.json'))]
    by = {}
    for r in rows:
        by.setdefault((r['system'], r['budget']), {})[r['question_id']] = r
    for system in SYSTEMS:
        for budget in BUDGETS:
            if len(by.get((system, budget), {})) != len(records):
                raise RuntimeError(f'incomplete matrix: {system}@{budget}')
    qids = [r['question_id'] for r in records]
    papers = [r['source'] for r in records]
    metric = 'official_qasper_evidence_f1'

    def per_question(system, key=metric):
        return [sum(float(by[(system, b)][q][key] or 0.0) for b in BUDGETS) / len(BUDGETS) for q in qids]

    agree, rap, leaf = per_question('agreement'), per_question('raptor'), per_question('all_leaf')
    # (name, diffs, kind): "ni" = non-inferiority at MARGIN, "sup" = one-sided superiority.
    family = [('H_a_agreement_noninferior_to_raptor', [a - r for a, r in zip(agree, rap)], 'ni'),
              ('H_b_agreement_superior_to_all_leaf', [a - l for a, l in zip(agree, leaf)], 'sup')]
    if SPLIT == 'test':
        family.insert(1, ('T2_all_leaf_noninferior_to_raptor', [l - r for l, r in zip(leaf, rap)], 'ni'))
    p_values = [non_inferiority(d, papers, MARGIN)['p'] if kind == 'ni'
                else cluster_sign_flip_test(d, papers, alternative='greater')
                for _, d, kind in family]
    adjusted = holm(p_values)
    primary = {'metric': 'official evidence F1, mean of 512/1024/2048 per question',
               'split': SPLIT, 'holm_family_size': len(family)}
    for (name, d, kind), p, p_adj in zip(family, p_values, adjusted):
        primary[name] = {'mean_diff': sum(d) / len(d), 'ci95': cluster_bootstrap_ci(d, papers),
                         'test': 'non-inferiority' if kind == 'ni' else 'superiority',
                         'margin': MARGIN if kind == 'ni' else None,
                         'p_one_sided': p, 'p_holm': p_adj, 'supported': p_adj < 0.05}
    keys = ['citation_f1', metric, 'answer_f1', 'gold_paragraph_char_coverage_best_ref',
            'packed_leaf_touch_recall', 'context_tokens', 'total_api_tokens']
    table = {}
    for (system, budget), items in by.items():
        values = list(items.values())
        table[f'{system}@{budget}'] = {k: sum(float(v[k] or 0.0) for v in values) / len(values) for k in keys}
    secondary = {}
    for budget in BUDGETS:
        for other in ('raptor', 'all_leaf'):
            for key in (metric, 'gold_paragraph_char_coverage_best_ref', 'citation_f1', 'answer_f1'):
                diffs = [float(by[('agreement', budget)][q][key] or 0) - float(by[(other, budget)][q][key] or 0) for q in qids]
                secondary[f'agreement-{other}@{budget}:{key}'] = {
                    'mean_diff': sum(diffs) / len(diffs), 'ci95': cluster_bootstrap_ci(diffs, papers)}
    prepared = [json.loads(p.read_text()) for p in sorted((root / 'prepared').glob('*.json'))]
    cost = {
        'raptor_index_seconds': json.loads((root / 'raptor_index_seconds.json').read_text())['seconds']
        if (root / 'raptor_index_seconds.json').exists() else None,
        'sbert_seconds_total': sum(p['seconds']['sbert'] for p in prepared),
        'raptor_search_seconds_total': sum(p['seconds']['raptor_search'] for p in prepared),
    }
    v9.dump(root / 'confirmation_results.json', {'primary': primary, 'table': table,
                                                'secondary': secondary, 'cost': cost,
                                                'questions': len(records), 'split': SPLIT,
                                                'test_accessed': SPLIT == 'test'})
    print(json.dumps(primary, indent=2))
    return primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'generate', 'verify', 'report'], required=True)
    parser.add_argument('--output', default='artifacts/v10_confirm')
    parser.add_argument('--config', default='artifacts/baselines/main/config.json')
    parser.add_argument('--api-config', help='local private config holding openai_api_key')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--split', choices=['confirm', 'test'], default='confirm')
    parser.add_argument('--open-test-once', action='store_true',
                        help='required with --split test: the preregistered one-shot test run')
    args = parser.parse_args()
    global SPLIT, VERSION
    if args.split == 'test':
        if not args.open_test_once:
            raise SystemExit('--split test requires --open-test-once (one-shot, preregistered)')
        SPLIT, VERSION = 'test', 'v10-test.1'
        if args.output == 'artifacts/v10_confirm':
            args.output = 'artifacts/v10_test'

    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    papers, records, hashes = load_manifest()
    if args.phase == 'report':
        report(root, records)
        return
    settings = replace(Settings.from_json(ROOT / args.config), llm_model='gpt-4o-mini',
                       llm_provider='openai', chunk_context='none',
                       parent_policy_checkpoint=None, section_policy_checkpoint=None)
    hierarchy = HierarchyBuilder(settings).build(documents_from_paper_records(papers))
    code = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        'scripts/run_v10_confirm.py', 'scripts/run_v9_directions.py',
        'src/edahr/agreement.py', 'src/edahr/experimental_v9.py')}
    protocol = {'version': VERSION, 'systems': SYSTEMS, 'budgets': BUDGETS, 'manifests': hashes,
                'settings': v9.safe_settings(settings), 'code_hashes': code,
                'margin': MARGIN, 'split': SPLIT, 'test_accessed': SPLIT == 'test'}
    path = root / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError('protocol changed since this run started; use a new output dir')
    else:
        v9.dump(path, protocol)
    if args.phase == 'prepare':
        prepare(settings, hierarchy, records, root)
        return
    prepared = [json.loads((root / 'prepared' / (v9.digest(r['question_id']) + '.json')).read_text())
                for r in records]
    import tiktoken
    encoder = tiktoken.get_encoding('cl100k_base')
    count = lambda text: len(encoder.encode(text))
    generator = verifier = None
    if args.phase == 'generate':
        key = json.loads((ROOT / args.api_config).read_text(encoding='utf-8-sig'))['openai_api_key']
        generator = v9.ExperimentGenerator(settings.llm_model, key, root / 'cache/generator')
    else:
        from edahr.models import NliVerifier
        verifier = v9.CachedVerifier(NliVerifier(settings.nli_model, settings.device), root / 'cache/verifier')
    jobs = []
    for budget in BUDGETS:
        for system in SYSTEMS:
            for record, data in zip(records, prepared):
                name = v9.digest([system, budget, record['question_id']]) + '.json'
                done, pending = root / 'rows' / name, root / 'pending' / name
                if done.exists() or (args.phase == 'generate' and pending.exists()):
                    continue
                if args.phase == 'verify' and not pending.exists():
                    continue
                context, pool, tokens = make_context(system, data, hierarchy, settings, budget, count)
                jobs.append((name, record, data, system, budget, context, pool, tokens))
    print(f'[{args.phase}] {len(jobs)} jobs', flush=True)

    def work(job):
        name, record, data, system, budget, context, pool, tokens = job
        if args.phase == 'generate':
            raw, usage, meta = generator.run(record['query'], context, 'standard')
            v9.dump(root / 'pending' / name, {'question_id': record['question_id'], 'system': system,
                    'budget': budget, 'raw': asdict(raw), 'usage': usage, 'meta': meta})
            return
        saved = json.loads((root / 'pending' / name).read_text())
        p = saved['raw']
        raw = Generation(p['answerable'], tuple(Claim(c['text'], tuple(c['citations']), c['confidence'])
                         for c in p['claims']), p['reason'], tuple(p['validation_errors']))
        row = v9.evaluate(record, data, system, budget, context, pool, tokens, raw, saved['usage'],
                          saved['meta'], hierarchy, verifier, settings)
        v9.dump(root / 'rows' / name, row)

    workers = args.workers if args.phase == 'generate' else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(work, jobs), 1):
            if index % 25 == 0:
                print(f'[{args.phase}] {index}/{len(jobs)}', flush=True)
    print(f'[{args.phase}] done', flush=True)


if __name__ == '__main__':
    main()
