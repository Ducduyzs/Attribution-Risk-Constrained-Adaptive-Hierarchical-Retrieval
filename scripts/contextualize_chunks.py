"""Generate and cache LLM chunk contexts (Anthropic Contextual Retrieval).

One call per leaf: the whole paper followed by the chunk, so the paper prefix
is shared across a paper's calls and OpenAI prompt caching applies. Output is
an append-only JSONL cache keyed by child id plus a hash of (model, paper
text, chunk) — resumable, and stale entries are rejected when applied.

  python scripts/contextualize_chunks.py --papers manifests/qasper_baseline_dev_papers.jsonl \
      --output artifacts/contextual/qasper_baseline_dev.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.config import Settings  # noqa: E402
from edahr.contextual import CONTEXT_PROMPT, context_key, document_text, load_contexts  # noqa: E402
from edahr.hierarchy import HierarchyBuilder  # noqa: E402
from edahr.qasper import documents_from_paper_records, read_jsonl  # noqa: E402
from edahr.text import token_estimate  # noqa: E402

# USD per 1M tokens for gpt-4o-mini (OpenAI list price; check before large runs).
PRICE_INPUT, PRICE_CACHED, PRICE_OUTPUT = 0.15, 0.075, 0.60


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings", default="artifacts/baselines/main/config.json",
                        help="chunking settings must match the benchmark run")
    parser.add_argument("--config", default="config.local.json", help="holds openai_api_key")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=150)
    parser.add_argument("--tpm", type=int, default=180_000,
                        help="client-side token-per-minute budget (account limit 200k)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads((PROJECT_ROOT / args.settings).read_text(encoding="utf-8"))
    fields = Settings.__dataclass_fields__
    settings = replace(Settings(), **{k: v for k, v in payload.items() if k in fields})
    hierarchy = HierarchyBuilder(settings).build(
        documents_from_paper_records(read_jsonl(PROJECT_ROOT / args.papers)))
    output = PROJECT_ROOT / args.output
    cached = load_contexts(output)

    documents: dict[str, str] = {}
    jobs = []
    for child_id in hierarchy.child_ids:
        node = hierarchy.node(child_id)
        document = documents.setdefault(node.source, document_text(hierarchy, node.source))
        key = context_key(args.model, document, node.text)
        if cached.get(child_id, {}).get("key") == key:
            continue
        jobs.append((child_id, node.source, document, node.text, key))

    # Project token estimate (approximate; actual usage is logged per call).
    doc_tokens = {s: token_estimate(d) for s, d in documents.items()}
    est_in = sum(doc_tokens[s] + token_estimate(c) + 80 for _, s, _, c, _ in jobs)
    est_first = sum(doc_tokens[s] for s in {s for _, s, _, _, _ in jobs})
    est_out = 80 * len(jobs)
    worst = est_in * PRICE_INPUT / 1e6 + est_out * PRICE_OUTPUT / 1e6
    best = ((est_in - est_first) * PRICE_CACHED + est_first * PRICE_INPUT) / 1e6 \
        + est_out * PRICE_OUTPUT / 1e6
    print(f"papers={len(documents)} leaves={len(hierarchy.child_ids)} "
          f"cached={len(hierarchy.child_ids) - len(jobs)} to_generate={len(jobs)}")
    print(f"estimated input tokens={est_in:,} output~{est_out:,}; "
          f"cost ${best:.2f} (prompt cache hits) .. ${worst:.2f} (no caching)")
    if args.dry_run or not jobs:
        return

    from openai import OpenAI
    key = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))["openai_api_key"]
    client = OpenAI(api_key=key)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    window: list[tuple[float, int]] = []  # (timestamp, tokens) sent in the last 60 s

    def throttle(tokens: int) -> None:
        while True:
            with lock:
                now = time.monotonic()
                while window and now - window[0][0] > 60.0:
                    window.pop(0)
                if sum(t for _, t in window) + tokens <= args.tpm:
                    window.append((now, tokens))
                    return
                wait = 60.0 - (now - window[0][0]) + 0.1
            time.sleep(max(0.1, wait))
    usage = {"prompt": 0, "cached": 0, "completion": 0, "done": 0}

    def run(job):
        child_id, source, document, chunk, key_hash = job
        throttle(doc_tokens[source] + token_estimate(chunk) + 200)
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=args.model, temperature=0.0,
                    max_tokens=args.max_output_tokens,
                    messages=[{"role": "user", "content": CONTEXT_PROMPT.format(
                        document=document, chunk=chunk)}],
                )
                break
            except Exception:  # noqa: BLE001 - rate limits / transient errors
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))
        details = getattr(response.usage, "prompt_tokens_details", None)
        record = {
            "child_id": child_id, "source": source, "key": key_hash,
            "model": args.model, "response_model": response.model,
            "context": response.choices[0].message.content.strip(),
            "prompt_tokens": response.usage.prompt_tokens,
            "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
            "completion_tokens": response.usage.completion_tokens,
        }
        with lock:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            usage["prompt"] += record["prompt_tokens"]
            usage["cached"] += record["cached_tokens"]
            usage["completion"] += record["completion_tokens"]
            usage["done"] += 1
            if usage["done"] % 50 == 0:
                print(f"  {usage['done']}/{len(jobs)}", flush=True)

    # Papers in sequence (first call warms the prompt cache), leaves in parallel.
    by_source: dict[str, list] = {}
    for job in jobs:
        by_source.setdefault(job[1], []).append(job)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for paper_jobs in by_source.values():
            run(paper_jobs[0])
            for future in as_completed([pool.submit(run, job) for job in paper_jobs[1:]]):
                future.result()
    cost = ((usage["prompt"] - usage["cached"]) * PRICE_INPUT + usage["cached"] * PRICE_CACHED
            + usage["completion"] * PRICE_OUTPUT) / 1e6
    print(f"done: prompt={usage['prompt']:,} (cached {usage['cached']:,}) "
          f"completion={usage['completion']:,} -> ${cost:.3f}")


if __name__ == "__main__":
    main()
