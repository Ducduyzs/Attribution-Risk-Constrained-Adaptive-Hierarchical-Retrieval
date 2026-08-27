# Reproduction commands

Run from the repository root in PowerShell. Never commit `config.local.json`
or API keys.

## 1. Offline verification

```powershell
D:\edahr_env\Scripts\python.exe -m pytest -q
```

Prepare stable paper/question manifests directly from official QASPER JSON:

```powershell
D:\edahr_env\Scripts\python.exe scripts\prepare_qasper.py
```

## 2. Fresh paper-scoped rollouts

For QASPER, use the structured full text embedded in the dataset (no PDF
download is required):

```powershell
D:\edahr_env\Scripts\python.exe scripts\run_qasper_rollouts.py `
  --split train --out data\rollouts_v5_train_fresh.jsonl `
  --config config.local.json --provider openai --model gpt-4o-mini
```

Use `--paper-limit 1 --question-limit 1 --max-groups 1` for a low-cost smoke run.
The generic PDF workflow remains available below.

The input JSONL must contain `question_id`, `query`, `source` (PDF filename),
`answer` or `gold_answer`, and preferably `gold_quotes`. The runner maps quotes
to leaf IDs and reports the evaluable rate.

```powershell
D:\edahr_env\Scripts\python.exe scripts\run_rollouts.py `
  --records data\manifests\train.jsonl `
  --pdf-dir data\raw_pdfs `
  --out data\rollouts_v5_train.jsonl `
  --config config.local.json `
  --samples 1
```

Repeat with the paper-disjoint dev manifest and a distinct output path.

## 3. Train independent v5 gates

The frozen v5 checkpoints use a random forest for the parent gate and gradient
boosting for the section gate. Training uses train only; estimator family and
threshold are selected on dev, never test.

```powershell
D:\edahr_env\Scripts\python.exe scripts\train_tree_policy.py `
  --train data\rollouts_v5_train_fresh.jsonl `
  --dev data\rollouts_v5_dev_fresh.jsonl `
  --out checkpoints\policy_parent_v5_final.joblib `
  --label parent --estimator rf --epsilon 0.02 --delta 0.05 --tau 0.02 --seed 42

D:\edahr_env\Scripts\python.exe scripts\train_tree_policy.py `
  --train data\rollouts_v5_train_fresh.jsonl `
  --dev data\rollouts_v5_dev_fresh.jsonl `
  --out checkpoints\policy_section_v5_final.joblib `
  --label section --estimator gb --epsilon 0.02 --delta 0.05 --tau 0.02 --seed 42
```

Each command creates a `.metadata.json` sidecar with checkpoint and rollout
hashes. Point `config.local.json` to both final checkpoints before evaluation.

## 4. Decompose benchmark artifacts

```powershell
D:\edahr_env\Scripts\python.exe scripts\decompose_drift.py `
  data\artifacts\artifacts_B_flat.jsonl `
  data\artifacts\artifacts_B_static.jsonl `
  data\artifacts\artifacts_prior.jsonl `
  data\artifacts\artifacts_learned.jsonl `
  --out-jsonl data\artifacts\v5_decomposition.jsonl `
  --out-csv data\artifacts\v5_decomposition.csv `
  --report analysis\v5_decomposition.md
```

Archive the config, manifests, metadata sidecars and decomposition report for
every result table. Do not compare systems produced from different question
sets, generator settings or verifier thresholds.

## 5. Frozen failure analysis

```powershell
D:\edahr_env\Scripts\python.exe scripts\analyze_benchmark_failures.py `
  data\artifacts\v5_final\artifacts_B_flat.jsonl `
  data\artifacts\v5_final\artifacts_B_static.jsonl `
  data\artifacts\v5_final\artifacts_prior.jsonl `
  data\artifacts\v5_final\artifacts_learned_v5.jsonl `
  --out-csv data\artifacts\v5_final\failure_cases.csv `
  --out-json data\artifacts\v5_final\failure_summary.json `
  --report analysis\v5_failure_analysis_fresh.md
```

## 6. SciFact out-of-domain evaluation

Download the official AllenAI SciFact release, then prepare the labeled dev
evidence manifests:

```powershell
D:\edahr_env\Scripts\python.exe scripts\prepare_scifact.py `
  --split dev --data-dir data\scifact\data --out-dir data\manifests

D:\edahr_env\Scripts\python.exe scripts\run_qasper_benchmark.py `
  --paper-manifest scifact_dev_papers.jsonl `
  --question-manifest scifact_dev_questions.jsonl `
  --dataset-name scifact --split-name dev-ood-frozen --questions 40 `
  --config config.local.json --provider openai --model gpt-4o-mini `
  --artifact-dir data\artifacts\v5_scifact_ood `
  --report analysis\v5_scifact_ood.md
```

This is a citation/rationale OOD benchmark. Do not interpret its answer F1 as
stance accuracy without adding and separately validating a normalized stance
output contract.
