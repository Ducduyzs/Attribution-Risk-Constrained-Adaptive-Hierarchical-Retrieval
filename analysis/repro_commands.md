# Reproduction commands

Run from the repository root in PowerShell. Never commit `config.local.json`
or API keys.

## 1. Offline verification

```powershell
D:\edahr_env\Scripts\python.exe -m pytest -q
```

## 2. Fresh paper-scoped rollouts

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

```powershell
D:\edahr_env\Scripts\python.exe scripts\train_policy.py `
  --rollouts data\rollouts_v5_train.jsonl `
  --out checkpoints\policy_parent_v5_final.ts `
  --label label_parent --v5 --epsilon 0.02 --delta 0.05 --tau 0.02 --seed 42

D:\edahr_env\Scripts\python.exe scripts\train_policy.py `
  --rollouts data\rollouts_v5_train.jsonl `
  --out checkpoints\policy_section_v5_final.ts `
  --label label_section --v5 --epsilon 0.02 --delta 0.05 --tau 0.02 --seed 42
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
