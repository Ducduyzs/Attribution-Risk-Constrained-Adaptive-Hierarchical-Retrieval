"""V11 robustness runs: frozen v10 arms on a new dataset and a second generator.

Preregistered in analysis/v11_protocol.md. Arms, budgets, packer, verifier,
scoring and the 3-hypothesis Holm family are exactly v10's
(scripts/run_v10_confirm.py, reused unchanged); only the data and the
generator vary:

  --dataset   qasper-confirm | qasper-test | peerqa
  --generator gpt-4o-mini (local API client) | qwen2.5-7b (GPU, no API)

Prepared rankings are reused from v10 for QASPER, so a generator change never
re-runs retrieval. PeerQA rankings are prepared on the GPU (RAPTOR trees etc.).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import run_v10_confirm as R  # noqa: E402
import run_v9_directions as v9  # noqa: E402
from edahr.config import Settings  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402
from edahr.schemas import Claim, Generation  # noqa: E402

GENERATORS = {'gpt-4o-mini': 'gpt-4o-mini', 'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct'}
LENGTH = '\nKeep the complete answer within 120 words. Avoid redundant claims.\n'
PEERQA_META = ROOT / 'manifests/peerqa_metadata.json'


class QwenExperimentGenerator:
    """Local Qwen with the v9 prompt + length instruction; content-addressed cache."""

    def __init__(self, model_name, folder, device='cuda'):
        from edahr.models import LocalStructuredGenerator
        self.local = LocalStructuredGenerator(model_name, device, max_new_tokens=700)
        self.model_name, self.folder = model_name, Path(folder)

    def run(self, query, context, mode='standard'):
        import torch
        from edahr.models import _generation_from_payload, _grounded_prompt, _invalid_generation, _json_payload
        if not context:
            return Generation(False, reason='No context fits budget.'), {'prompt_tokens': 0, 'completion_tokens': 0}, {}
        ids = [b.context_id for b in context]
        prompt = _grounded_prompt(query, context, json_only=True) + LENGTH
        key = v9.digest([R.VERSION, self.model_name, prompt, 700])
        path = self.folder / (key + '.json')
        cached = path.exists()
        if cached:
            data = json.loads(path.read_text())
        else:
            messages = [{'role': 'system', 'content': 'Answer with a single JSON object only. No prose.'},
                        {'role': 'user', 'content': prompt}]
            inputs = self.local.tokenizer.apply_chat_template(
                messages, return_tensors='pt', add_generation_prompt=True, return_dict=True)
            input_ids = inputs['input_ids'].to(self.local.device)
            mask = inputs.get('attention_mask')
            mask = mask.to(self.local.device) if mask is not None else None
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = self.local.model.generate(input_ids, attention_mask=mask,
                                                   max_new_tokens=700, do_sample=False)
            new = output[0][input_ids.shape[1]:]
            data = {'text': self.local.tokenizer.decode(new, skip_special_tokens=True),
                    'usage': {'prompt_tokens': int(input_ids.shape[1]), 'completion_tokens': int(len(new)),
                              'total_tokens': int(input_ids.shape[1] + len(new))},
                    'seconds': time.perf_counter() - t0}
            v9.dump(path, data)
        try:
            raw = _generation_from_payload(_json_payload(data['text']), ids)
        except (TypeError, ValueError) as exc:
            raw = _invalid_generation('local_invalid_json', str(exc)[:300])
        return raw, data['usage'], {'cached': cached, 'cache_key': key, 'seconds': data['seconds'],
                                    'model_returned': self.model_name}


def load_dataset(name):
    if name == 'peerqa':
        meta = json.loads(PEERQA_META.read_text(encoding='utf-8'))
        hashes = {}
        for file, expected in meta['sha256'].items():
            actual = hashlib.sha256((ROOT / 'manifests' / file).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'PeerQA manifest hash mismatch: {file}')
            hashes[file] = actual
        return (read_jsonl(ROOT / 'manifests/peerqa_papers.jsonl'),
                read_jsonl(ROOT / 'manifests/peerqa_questions.jsonl'), hashes)
    R.SPLIT = 'test' if name == 'qasper-test' else 'confirm'
    return R.load_manifest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['qasper-confirm', 'qasper-test', 'peerqa'], required=True)
    parser.add_argument('--generator', choices=sorted(GENERATORS), required=True)
    parser.add_argument('--phase', choices=['prepare', 'generate', 'verify', 'report'], required=True)
    parser.add_argument('--config', default='artifacts/baselines/main/config.json')
    parser.add_argument('--api-config')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = ROOT / 'artifacts' / f"v11_{args.dataset}_{args.generator.replace('.', '')}"
    root.mkdir(parents=True, exist_ok=True)
    papers, records, hashes = load_dataset(args.dataset)
    # The family/version used by R.report: QASPER confirm keeps its 2-hypothesis
    # family; test and PeerQA use the 3-hypothesis test family.
    R.SPLIT = 'confirm' if args.dataset == 'qasper-confirm' else 'test'
    R.VERSION = 'v11.1'
    if args.phase == 'report':
        R.report(root, records)
        return
    settings = replace(Settings.from_json(ROOT / args.config), llm_model=GENERATORS[args.generator],
                       llm_provider='openai' if args.generator == 'gpt-4o-mini' else 'local',
                       chunk_context='none', parent_policy_checkpoint=None, section_policy_checkpoint=None)
    hierarchy = HierarchyBuilder(settings).build(documents_from_paper_records(papers))
    code = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        'scripts/run_v11.py', 'scripts/run_v10_confirm.py', 'scripts/run_v9_directions.py',
        'src/edahr/agreement.py', 'src/edahr/experimental_v9.py', 'src/edahr/peerqa.py')}
    protocol = {'version': R.VERSION, 'dataset': args.dataset, 'generator': GENERATORS[args.generator],
                'systems': R.SYSTEMS, 'budgets': R.BUDGETS, 'manifests': hashes,
                'settings': v9.safe_settings(settings), 'code_hashes': code, 'margin': R.MARGIN}
    path = root / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('protocol changed since this run started; use a new output dir')
    v9.dump(path, protocol)
    reuse = {'qasper-confirm': 'artifacts/v10_confirm', 'qasper-test': 'artifacts/v10_test'}.get(args.dataset)
    if reuse and not (root / 'prepared').exists():
        shutil.copytree(ROOT / reuse / 'prepared', root / 'prepared')
    if args.phase == 'prepare':
        if reuse:
            print('[prepare] reused v10 rankings', flush=True)
            return
        R.prepare(settings, hierarchy, records, root)
        return
    prepared = [json.loads((root / 'prepared' / (v9.digest(r['question_id']) + '.json')).read_text())
                for r in records]
    import tiktoken
    encoder = tiktoken.get_encoding('cl100k_base')
    count = lambda text: len(encoder.encode(text))
    if args.phase == 'generate':
        if args.generator == 'gpt-4o-mini':
            key = json.loads((ROOT / args.api_config).read_text(encoding='utf-8-sig'))['openai_api_key']
            generator = v9.ExperimentGenerator('gpt-4o-mini', key, root / 'cache/generator')
        else:
            generator = QwenExperimentGenerator(GENERATORS[args.generator], root / 'cache/generator', settings.device)
    else:
        from edahr.models import NliVerifier
        verifier = v9.CachedVerifier(NliVerifier(settings.nli_model, settings.device), root / 'cache/verifier')
    jobs = []
    for budget in R.BUDGETS:
        for system in R.SYSTEMS:
            for record, data in zip(records, prepared):
                name = v9.digest([system, budget, record['question_id']]) + '.json'
                done, pending = root / 'rows' / name, root / 'pending' / name
                if done.exists() or (args.phase == 'generate' and pending.exists()):
                    continue
                if args.phase == 'verify' and not pending.exists():
                    continue
                jobs.append((name, record, data, system, budget,
                             *R.make_context(system, data, hierarchy, settings, budget, count)))
    print(f'[{args.phase}] {len(jobs)} jobs', flush=True)
    workers = args.workers if (args.phase == 'generate' and args.generator == 'gpt-4o-mini') else 1
    from concurrent.futures import ThreadPoolExecutor

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
        v9.dump(root / 'rows' / name, v9.evaluate(record, data, system, budget, context, pool, tokens,
                raw, saved['usage'], saved['meta'], hierarchy, verifier, settings))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, _ in enumerate(executor.map(work, jobs), 1):
            if index % 25 == 0:
                print(f'[{args.phase}] {index}/{len(jobs)}', flush=True)
    print(f'[{args.phase}] done', flush=True)


if __name__ == '__main__':
    main()
