"""Run every v9 direction on the DEV split only, with resumable caches.

Two phases: prepare all GPU rankings, then generate/verify at matched budgets.
No test manifest can be passed to this runner. Gold is only used in scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from edahr.baselines import auto_label_gold_children, raptor_faithful_retriever
from edahr.config import Settings
from edahr.context import assemble_context
from edahr.contextual import apply_chunk_context
from edahr.evaluation import aggregate, citation_f1, qasper_answer_token_f1, qasper_evidence_f1
from edahr.experimental_v9 import (Unit, blocks_from_units, dependency_groups, diagnostics,
    leaf_units, restrict_to_visible, select_units, sentence_units, verify_visible)
from edahr.hierarchy import HierarchyBuilder
from edahr.models import (BGEM3Encoder, BGEReranker, NliVerifier, OpenAIStructuredGenerator,
    _generation_from_payload, _generation_schema, _grounded_prompt)
from edahr.pipeline import classify_query
from edahr.qasper import documents_from_paper_records, read_jsonl
from edahr.schemas import ContextBlock, Generation, Claim, Level
from edahr.statistics import compare, holm

SYSTEMS = ['flat24', 'all_leaf', 'contextual24', 'contextual_all', 'mmr', 'coverage',
           'sentence', 'window', 'dependency', 'evidence_first', 'dependency_first', 'raptor']
VERSION = 'v9.1'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_settings(settings):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not any(x in k.lower() for x in ('api_key', 'password', 'secret'))}
        return value
    return clean(asdict(settings))


class CachedReranker:
    def __init__(self, reranker, folder):
        self.reranker, self.folder = reranker, Path(folder)
        self.calls = 0
        self.texts = 0

    def score(self, query, texts):
        key = digest([VERSION, query, texts])
        path = self.folder / (key + '.json')
        if path.exists():
            return json.loads(path.read_text())['scores']
        t0 = time.perf_counter()
        scores = []
        for start in range(0, len(texts), 16):
            scores.extend(self.reranker.score(query, texts[start:start+16]))
        dump(path, {'scores': scores, 'seconds': time.perf_counter() - t0, 'texts': len(texts)})
        self.calls += 1
        self.texts += len(texts)
        return scores


class ExperimentGenerator(OpenAIStructuredGenerator):
    def __init__(self, model, api_key, folder):
        super().__init__(model, api_key)
        self.folder = Path(folder)
        self.client = self.client.with_options(timeout=90, max_retries=2)

    def run(self, query, context, mode='standard'):
        if not context:
            return Generation(False, reason='No context fits budget.'), {'prompt_tokens': 0, 'completion_tokens': 0}, {}
        ids = [b.context_id for b in context]
        prompt = _grounded_prompt(query, context)
        # Identical answer-length constraint for both generation conditions.
        prompt += '\nKeep the complete answer within 120 words. Avoid redundant claims.\n'
        schema = _generation_schema(ids)
        if mode == 'evidence_first':
            prompt += ('First list the information requirements in the question and map each to supplied context IDs. '
                       'Mark unsupported requirements with an empty list. Then answer using this evidence plan. '
                       'Do not invent extra requirements or claims merely to cite more passages. '
                       'Return evidence_plan before the answer fields.\n')
            props = schema['schema']['properties']
            schema['schema']['properties'] = {'evidence_plan': {
                'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                    'required': ['requirement', 'citations'], 'properties': {
                        'requirement': {'type': 'string'},
                        'citations': {'type': 'array', 'items': {'type': 'string', 'enum': ids}}}}}, **props}
            schema['schema']['required'] = ['evidence_plan', *schema['schema']['required']]
        key = digest([VERSION, self.model_name, prompt, schema, 700])
        path = self.folder / (key + '.json')
        cached = path.exists()
        if cached:
            data = json.loads(path.read_text())
        else:
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}], temperature=0,
                max_tokens=700, response_format={'type': 'json_schema', 'json_schema': schema})
            choice = response.choices[0]
            if choice.finish_reason != 'stop' or getattr(choice.message, 'refusal', None):
                raise RuntimeError('Generation incomplete/refused: ' + str(choice.finish_reason))
            payload = json.loads(choice.message.content)
            data = {'payload': payload, 'usage': response.usage.model_dump(),
                    'seconds': time.perf_counter() - t0, 'model_returned': response.model,
                    'system_fingerprint': response.system_fingerprint}
            dump(path, data)
        raw = _generation_from_payload(data['payload'], ids)
        return raw, data['usage'], {'cached': cached, 'cache_key': key,
                'seconds': data['seconds'], 'model_returned': data['model_returned'],
                'system_fingerprint': data.get('system_fingerprint'),
                'evidence_plan': data['payload'].get('evidence_plan', [])}


class CachedVerifier:
    def __init__(self, verifier, folder):
        self.verifier, self.folder = verifier, Path(folder)

    def score_details(self, claim, evidence):
        path = self.folder / (digest([VERSION, claim, evidence]) + '.json')
        if path.exists():
            return tuple(json.loads(path.read_text()))
        scores = self.verifier.score_details(claim, evidence)
        dump(path, scores)
        return scores


def load_dev(args):
    paths = [ROOT / 'manifests/qasper_baseline_dev_papers.jsonl',
             ROOT / 'manifests/qasper_baseline_dev_questions.jsonl']
    papers, records = [read_jsonl(path) for path in paths]
    records = [r for r in records if r.get('gold_quotes')][:args.questions]
    sources = {str(r['paper_id']) for r in records}
    papers = [p for p in papers if str(p['paper_id']) in sources]
    return papers, records, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(args, settings, hierarchy, records, root):
    from edahr.index import MultiRepresentationIndex
    paths = [root / 'prepared' / (digest(r['question_id']) + '.json') for r in records]
    if all(p.exists() for p in paths):
        return [json.loads(p.read_text()) for p in paths]
    print('[prepare] loading GPU embedding and reranker', flush=True)
    encoder = BGEM3Encoder(settings.embedding_model, settings.device, settings.use_fp16)
    index = MultiRepresentationIndex(hierarchy, encoder, settings)
    reranker = CachedReranker(BGEReranker(settings.reranker_model, settings.device, settings.use_fp16), root / 'cache/reranker')
    contexts_path = ROOT / 'artifacts/contextual/qasper_baseline_dev.jsonl'
    contextual = apply_chunk_context(hierarchy, 'llm', contexts_path)
    print('[prepare] loading RAPTOR trees', flush=True)
    raptor = raptor_faithful_retriever(hierarchy, settings)
    output = []
    for n, (record, path) in enumerate(zip(records, paths), 1):
        if path.exists():
            output.append(json.loads(path.read_text()))
            continue
        t0 = time.perf_counter()
        query, source = record['query'], record['source']
        leaves = [cid for cid in hierarchy.child_ids if hierarchy.node(cid).source == source]
        initial = index.search(query, settings.candidate_k, source=source)
        initial_ids = [hit.node_id for hit in initial]
        raw_scores = reranker.score(query, [hierarchy.node(cid).text for cid in leaves])
        contextual_scores = reranker.score(query, [contextual.node(cid).embedding_text for cid in leaves])
        raw = dict(zip(leaves, raw_scores))
        ctx = dict(zip(leaves, contextual_scores))
        pool = initial_ids[:settings.rerank_k]
        raptor_pool = [h.node_id for h in raptor.search(query, settings.candidate_k, source=source)][:settings.rerank_k]
        rank = lambda ids, scores: sorted([(cid, scores[cid]) for cid in ids], key=lambda x: -x[1])
        units = sentence_units(hierarchy, leaves)
        sentence_scores = reranker.score(query, [u.text for u in units])
        units = [replace(u, score=s) for u, s in zip(units, sentence_scores)]
        data = {'question_id': record['question_id'], 'source': source, 'query': query,
                'rankings': {'flat24': rank(pool, raw), 'all_leaf': rank(leaves, raw),
                             'contextual24': rank(pool, ctx), 'contextual_all': rank(leaves, ctx),
                             'raptor': rank(raptor_pool, raw)},
                'initial_ids': initial_ids, 'sentences': [asdict(u) for u in units],
                'prepare_seconds': time.perf_counter() - t0, 'leaf_count': len(leaves)}
        dump(path, data)
        output.append(data)
        print(f'[prepare] {n}/{len(records)} leaves={len(leaves)} sentences={len(units)} seconds={data["prepare_seconds"]:.1f}', flush=True)
    return output


def make_context(system, prepared, hierarchy, settings, budget, count):
    base = system if system in prepared['rankings'] else 'all_leaf'
    ranking = prepared['rankings'][base]
    pool = [cid for cid, _ in ranking]
    if system in ('flat24', 'all_leaf', 'contextual24', 'contextual_all', 'raptor', 'evidence_first'):
        # Existing packer as baseline; correct model-token cap after packing.
        candidates = assemble_context(hierarchy, dict(ranking), classify_query(prepared['query']),
            replace(settings, context_token_budget=budget))
        units = [Unit(b.node_id, b.text, b.node_id, hierarchy.node(b.node_id).section_id or '',
                      b.char_start, b.char_end, b.utility) for b in candidates]
        chosen = select_units(units, prepared['query'], budget, count)
    elif system in ('mmr', 'coverage'):
        chosen = select_units(leaf_units(hierarchy, ranking), prepared['query'], budget, count,
                              method=system)
    else:
        units = [Unit(**u) for u in prepared['sentences']]
        mode = 'dependency' if system in ('dependency', 'dependency_first') else system
        groups = dependency_groups(units, mode)
        chosen = select_units(units, prepared['query'], budget, count,
            method='top' if system == 'sentence' else 'coverage', groups=groups)
    context = blocks_from_units(chosen, hierarchy, count)
    # All arms share the same exact content+source-header cap.
    def cost(blocks):
        return count('\n\n'.join(f'[{b.context_id}] {b.source}, pages {b.page_start}-{b.page_end}\n{b.text}' for b in blocks))
    if cost(context) > budget:
        raise AssertionError(f'Context header allowance too small: {cost(context)} > {budget}')
    return context, pool, cost(context)


def evaluate(record, prepared, system, budget, context, pool, tokens, raw, usage, genmeta,
             hierarchy, verifier, settings):
    gold, _ = auto_label_gold_children(hierarchy, record)
    t0 = time.perf_counter()
    verified, evidence, trace = verify_visible(raw, context, hierarchy, verifier, settings, set(pool))
    visible = restrict_to_visible(hierarchy, context)
    paragraphs = {}
    for e in evidence.values():
        # Resolve only the quoted fragment's paragraphs, not all leaf paragraphs.
        for pid, text in hierarchy.node(e.node_id).metadata.get('paragraph_texts', {}).items():
            from edahr.text import normalize
            if normalize(e.quote) in normalize(text) or normalize(text) in normalize(e.quote):
                paragraphs[pid] = text
            else:
                # Multi-sentence quoted leaf may straddle paragraph boundaries.
                from edahr.text import sentences
                if any(normalize(s) in normalize(text) for s in sentences(e.quote)):
                    paragraphs[pid] = text
    answer = ' '.join(c.text for c in verified.claims) or 'Unanswerable'
    raw_answer = ' '.join(c.text for c in raw.claims) or 'Unanswerable'
    d = diagnostics(record, hierarchy, pool, context, raw, evidence, gold)
    row = {'system': system, 'budget': budget, 'question_id': record['question_id'], 'source': record['source'],
        'query': record['query'], 'predicted_answer': answer, 'raw_answer': raw_answer,
        'answer_f1': qasper_answer_token_f1(answer, record['reference_answers']),
        'raw_answer_f1': qasper_answer_token_f1(raw_answer, record['reference_answers']),
        'citation_f1': citation_f1({e.node_id for e in evidence.values()}, gold) if gold else None,
        'official_qasper_evidence_f1': qasper_evidence_f1(list(paragraphs.values()), record['reference_evidence_sets']),
        'predicted_evidence_texts': list(paragraphs.values()), 'reference_evidence_sets': record['reference_evidence_sets'],
        'reference_answers': record['reference_answers'],
        'context_tokens': tokens, 'context_content_tokens': sum(b.token_count for b in context),
        'prompt_tokens': usage.get('prompt_tokens', 0), 'completion_tokens': usage.get('completion_tokens', 0),
        'total_api_tokens': usage.get('total_tokens', 0),
        'generated_claim_count': len(raw.claims), 'verified_claim_count': len(verified.claims),
        'claim_acceptance_proxy': len(verified.claims) / max(1, len(raw.claims)),
        'answer_words': len(answer.split()), 'over_answer_word_limit': len(answer.split()) > 120,
        'generation': asdict(raw), 'verified_generation': asdict(verified),
        'verification_trace': trace, 'evidence': [asdict(e) for e in evidence.values()],
        'generation_metadata': genmeta, 'verification_seconds': time.perf_counter() - t0,
        'gold_child_ids': sorted(gold), 'ranking': prepared['rankings'].get(system, prepared['rankings']['all_leaf']),
        **d}
    return row


def write_report(root, records, statistics=True):
    rows = [json.loads(p.read_text()) for p in sorted((root / 'rows').glob('*.json'))]
    groups = {}
    for row in rows:
        groups.setdefault((row['system'], row['budget']), []).append(row)
    summaries = {f'{s}@{b}': {'n': len(items), **aggregate([
        {k: v for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)} for row in items])}
        for (s, b), items in groups.items()}
    dump(root / 'summaries.json', summaries)
    for (system, budget), items in groups.items():
        path = root / f'artifacts_{system}_{budget}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items), encoding='utf-8')
    lines = ['# V9 directions — exploratory DEV only', '',
        'Exact cl100k_base context+headers budgets; shared GPT-4o-mini and visible-fragment verifier.',
        'Scores are not directly interchangeable with v8 (tokenizer, length instruction, visible verification).',
        'All methods are heuristic prototypes, not validated novelty claims. Test split untouched.', '',
        '| System | Budget | N | Leaf F1 | Official evidence F1 | Answer F1 | Raw answer F1 | Context tokens | API tokens |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in summaries.items():
        system, budget = name.split('@')
        lines.append(f'| {system} | {budget} | {s["n"]} | {s.get("citation_f1", 0):.4f} | {s["official_qasper_evidence_f1"]:.4f} | {s["answer_f1"]:.4f} | {s["raw_answer_f1"]:.4f} | {s["context_tokens"]:.1f} | {s["total_api_tokens"]:.1f} |')
    comparisons = []
    clusters = {r['question_id']: r['source'] for r in records}
    for (system, budget), items in (groups.items() if statistics else []):
        if system == 'raptor' or ('raptor', budget) not in groups:
            continue
        baseline = groups[('raptor', budget)]
        if len(items) != len(records) or len(baseline) != len(records):
            continue
        for metric in ('citation_f1', 'official_qasper_evidence_f1', 'answer_f1'):
            outcome = compare({r['question_id']: r.get(metric) for r in items},
                              {r['question_id']: r.get(metric) for r in baseline}, clusters)
            comparisons.append({'system': system, 'budget': budget, 'metric': metric, **outcome})
    for result, adjusted in zip(comparisons, holm([r['p'] for r in comparisons])):
        result['p_holm'] = adjusted
    if statistics:
        dump(root / 'comparisons_vs_raptor.json', comparisons)
    lines += ['', '## Stage diagnostics', '',
        '| System@budget | Pool gold recall | Packed leaf touch | Gold paragraph character coverage | Raw citation recall | Verified citation recall |',
        '|---|---:|---:|---:|---:|---:|']
    for name, s in summaries.items():
        lines.append(f'| {name} | {s["pool_gold_recall"]:.3f} | {s["packed_leaf_touch_recall"]:.3f} | {s["gold_paragraph_char_coverage_best_ref"]:.3f} | {s["raw_cited_leaf_recall"]:.3f} | {s["verified_leaf_recall"]:.3f} |')
    lines += ['', 'Leaf touch is NOT full text retention. Character coverage is conservative exact-span coverage, not semantic sufficiency.',
              'Claim acceptance uses the same NLI verifier and is NOT an independent attribution quality measurement.',
              'Compute/setup and contextual/RAPTOR precomputation costs are separate from online API usage.',
              'Paired cluster tests are exploratory; Holm family includes every completed method/budget/metric vs RAPTOR.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Blinded, deterministic sample for later human claim-support audit; labels unfilled.
    audit = []
    for row in rows:
        for index, claim in enumerate(row['generation']['claims']):
            if int(digest([row['question_id'], row['system'], row['budget'], index])[:8], 16) % 17 == 0:
                audit.append({'audit_id': digest([row['question_id'], row['system'], row['budget'], index]),
                    'question': row['query'], 'claim': claim['text'],
                    'cited_text': [b['text'] for b in row['context'] if b['context_id'] in claim['citations']],
                    'supported': None, 'missing_condition': None, 'notes': ''})
    dump(root / 'human_audit_unlabelled.json', sorted(audit, key=lambda x: x['audit_id']))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='artifacts/baselines/main/config.json')
    parser.add_argument('--output', default='artifacts/v9_directions')
    parser.add_argument('--questions', type=int, default=75)
    parser.add_argument('--budgets', nargs='+', type=int, default=[512, 1024, 2048])
    parser.add_argument('--systems', nargs='+', choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--phase', choices=['prepare', 'run', 'generate', 'verify', 'report'], default='run')
    parser.add_argument('--api-config', help='Local private config: only openai_api_key is read; never exported')
    parser.add_argument('--prepared-from', help='Reuse completed preparer output after manifest/settings validation')
    args = parser.parse_args()
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    papers, records, hashes = load_dev(args)
    if args.phase == 'report':
        write_report(root, records)
        return
    settings = replace(Settings.from_json(ROOT / args.config), llm_model='gpt-4o-mini',
        llm_provider='openai', chunk_context='none', parent_policy_checkpoint=None, section_policy_checkpoint=None)
    hierarchy = HierarchyBuilder(settings).build(documents_from_paper_records(papers))
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__), ROOT / 'src/edahr/experimental_v9.py']}
    protocol = {'version': VERSION, 'split': 'dev-exploratory', 'questions': len(records),
                'papers': len(papers), 'manifests': hashes, 'settings': safe_settings(settings),
                'code_hashes': code_hashes, 'tokenizer': 'cl100k_base', 'test_accessed': False}
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old != protocol:
            raise RuntimeError('Protocol changed; use a new output directory (no silent mixed provenance).')
    else:
        dump(protocol_path, protocol)
    if args.prepared_from and not (root / 'prepared').exists():
        import shutil
        source = ROOT / args.prepared_from
        previous = json.loads((source / 'protocol.json').read_text())
        for key in ('version', 'manifests', 'settings', 'questions', 'papers'):
            if previous[key] != protocol[key]:
                raise RuntimeError('Prepared cache incompatible: ' + key)
        shutil.copytree(source / 'prepared', root / 'prepared')
        dump(root / 'prepared_source.json', previous)
    if args.phase in ('generate', 'verify'):
        paths = [root / 'prepared' / (digest(r['question_id']) + '.json') for r in records]
        if not all(p.exists() for p in paths):
            raise RuntimeError('Prepare all GPU rankings first; split phases never load the GPU retriever.')
        prepared = [json.loads(p.read_text()) for p in paths]
    else:
        prepared = prepare(args, settings, hierarchy, records, root)
    if args.phase == 'prepare':
        return
    import tiktoken
    tokenizer = tiktoken.get_encoding('cl100k_base')
    count = lambda text: len(tokenizer.encode(text))
    generator = None
    verifier = None
    if args.phase != 'verify':
        api_key = settings.openai_api_key
        if args.api_config:
            api_key = json.loads((ROOT / args.api_config).read_text(encoding='utf-8-sig'))['openai_api_key']
        generator = ExperimentGenerator(settings.llm_model, api_key, root / 'cache/generator')
    if args.phase != 'generate':
        verifier = CachedVerifier(NliVerifier(settings.nli_model, settings.device), root / 'cache/verifier')
    jobs = []
    for budget in args.budgets:
        for system in args.systems:
            for record, data in zip(records, prepared):
                path = root / 'rows' / (digest([system, budget, record['question_id']]) + '.json')
                if path.exists():
                    continue
                pending = root / 'pending' / path.name
                if args.phase == 'generate' and pending.exists():
                    continue
                if args.phase == 'verify' and not pending.exists():
                    continue
                context, pool, tokens = make_context(system, data, hierarchy, settings, budget, count)
                jobs.append((path, record, data, system, budget, context, pool, tokens))
    print(f'[run] {len(jobs)} remaining jobs, workers={args.workers}', flush=True)
    def generate(job):
        pending = root / 'pending' / job[0].name
        if pending.exists():
            saved = json.loads(pending.read_text())
            payload = saved['raw']
            raw = Generation(payload['answerable'], tuple(Claim(c['text'], tuple(c['citations']), c['confidence']) for c in payload['claims']),
                             payload['reason'], tuple(payload['validation_errors']))
            return job, raw, saved['usage'], saved['meta']
        raw, usage, meta = generator.run(job[1]['query'], job[5],
             'evidence_first' if job[3] in ('evidence_first', 'dependency_first') else 'standard')
        return job, raw, usage, meta
    # Bounded batches avoid submitting thousands of paid requests before a failure is seen.
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(jobs), args.workers):
            for job, raw, usage, meta in executor.map(generate, jobs[start:start+args.workers]):
                path, record, data, system, budget, context, pool, tokens = job
                if args.phase == 'generate':
                    dump(root / 'pending' / path.name, {'question_id': record['question_id'], 'system': system,
                        'budget': budget, 'raw': asdict(raw), 'usage': usage, 'meta': meta})
                    done += 1
                    if done % 10 == 0:
                        print(f'[generate] {done}/{len(jobs)} {system}@{budget}', flush=True)
                    continue
                row = evaluate(record, data, system, budget, context, pool, tokens, raw, usage, meta,
                               hierarchy, verifier, settings)
                dump(path, row)
                done += 1
                if done % 10 == 0:
                    print(f'[run] {done}/{len(jobs)} {system}@{budget} q={record["question_id"]} F1={row["citation_f1"]}', flush=True)
                if done % 75 == 0:
                    write_report(root, records, statistics=False)
    if args.phase == 'generate':
        dump(root / 'generation_completed.json', {'pending': len(list((root / 'pending').glob('*.json'))),
             'questions': len(records), 'systems': args.systems, 'budgets': args.budgets})
        print('[done] generation; GPU verification remains', flush=True)
        return
    write_report(root, records)
    dump(root / 'completed.json', {'rows': len(list((root / 'rows').glob('*.json'))),
        'systems': args.systems, 'budgets': args.budgets, 'questions': len(records), 'test_accessed': False})
    print('[done] ' + str(root / 'report.md'), flush=True)


if __name__ == '__main__':
    main()
