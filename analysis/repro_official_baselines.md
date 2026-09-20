# Reproduction commands — faithful RAPTOR/LongRAG baselines

All commands run from the repository root. GPU machine used for the
reference runs: Vast.ai RTX 3090 24GB, `/venv/main` (torch 2.11+cu128,
Python 3.12). Local equivalent: CUDA GPU ≥12GB (16GB recommended).

API keys are never on the command line: `config.local.json` (gitignored)
or `OPENAI_API_KEY` / `GEMINI_API_KEY` env vars.

## 1. Install dependencies

```powershell
pip install torch transformers sentence-transformers FlagEmbedding faiss-cpu scikit-learn umap-learn google-genai openai tiktoken orjson
# Pinned versions: requirements.lock.txt (utf-16). New pins vs v7:
# umap-learn==0.5.12, pynndescent==0.6.0 (pyproject.toml updated).
pip install -e .
```

## 2. Prepare/download models (auto-download on first use; prefetch here)

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')"
python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/bart-large-cnn'); AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-large-en-v1.5'); snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
# Shared stack (reranker/NLI): BAAI/bge-reranker-v2-m3, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

## 3. Create frozen manifest

```powershell
python scripts/create_baseline_manifest.py --dev-papers 30 --dev-q-per-paper 3 --test-papers 60 --test-q-per-paper 2 --seed 42
# -> manifests/qasper_baseline_{dev,test}_{papers,questions}.jsonl + *_manifest_metadata.json (SHA-256, seed, commit)
```

## 4. Build RAPTOR index (per-paper trees, cached)

```powershell
# Runs inside smoke/main automatically; cache: artifacts/baselines/raptor/index/*.pkl
# RAPTOR config block: config.local.json -> raptor_faithful (provider local/openai/gemini)
```

## 5. Build LongRAG index (document units + bge-large-en-v1.5)

```powershell
# Runs inside smoke/main automatically; cache: artifacts/baselines/longrag/index/*.pkl
```

## 6. Smoke test (3 papers / 6 questions)

```powershell
python scripts/run_faithful_smoke.py --papers 3 --config config.local.json
# -> artifacts/baselines/{raptor,longrag}/smoke/ (config, run_metadata,
#    per_query_predictions, retrieval_traces, summary_metrics, errors, index_metadata)
```

## 7. Dev evaluation (30 papers / 75 questions, 3 phases)

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python scripts/run_faithful_main.py --phase faithful --config config.local.json
python scripts/run_faithful_main.py --phase flat --config config.local.json
python scripts/run_faithful_main.py --phase reader --config config.local.json
# -> artifacts/baselines/main/
```

## 8. Frozen main evaluation (test manifest — only after protocol lock)

```powershell
# Same script with --manifest test (flag to be added); do NOT run before Task 9 sign-off.
```

## 9. Result tables

```powershell
python scripts/run_faithful_main.py --phase compare
# -> artifacts/baselines/main/comparison.json + report.md
```

## 10. Tests

```powershell
python -m pytest tests/test_baselines.py tests/test_raptor_faithful.py tests/test_longrag_faithful.py tests/test_evaluation.py tests/test_qasper.py tests/test_pipeline.py -q
python -m pytest tests/ -q   # full suite
# Faithful factory builds needing models/API: set EDAHR_FAITHFUL_SMOKE=1 (else skipped with reason).
```

## Remote (Vast.ai) notes

- SSH upload limit through the Vast proxy is ~1MB per transfer: split files
  (`split -b 400k`) and reassemble with `cat`, verify SHA-256.
- Run long jobs detached: `setsid nohup python ... > log 2>&1 < /dev/null &`.
- Nothing on the instance survives recycle/destroy without a host volume:
  download `artifacts/baselines/**` after every run.
