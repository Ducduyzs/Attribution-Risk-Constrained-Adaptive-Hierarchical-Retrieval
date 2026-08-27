# Fresh V5 Failure Analysis

This is a diagnostic analysis of the frozen 40-paper test. It must not be used
to tune thresholds or select a replacement model.

`post_context_no_evidence` means gold reached the context candidate set but no verified evidence survived. These artifacts predate claim counters, so empty generation cannot be separated from verifier rejection.

## Input provenance

- `data/artifacts/v5_final/artifacts_B_flat.jsonl`: 40 rows, SHA-256 `91f23f873b7075db8ac9aa8c7e4d3d15b7eba673988a78c82867cde46a7445e2`
- `data/artifacts/v5_final/artifacts_B_static.jsonl`: 40 rows, SHA-256 `3d658911cb7c02805067f23810a3a95958d90fd6275c915ce0426a9e26b8b5cc`
- `data/artifacts/v5_final/artifacts_prior.jsonl`: 40 rows, SHA-256 `5952c7084f3240973f2840650d630c785d548b7546c5e5fddd456a1b2eb41a79`
- `data/artifacts/v5_final/artifacts_learned_v5.jsonl`: 40 rows, SHA-256 `6a2a9c8b855a9648ba7482a4a00525d54de05a395fb8bd536677e3532beba6e6`

## Failure counts

| system | candidate_selection_miss | full_grounding_success | partial_grounding_success | post_context_no_evidence | retrieval_miss_at_10 | total |
|---|---:|---:|---:|---:|---:|---:|
| B_flat | 0 | 1 | 0 | 37 | 2 | 40 |
| B_static | 0 | 1 | 4 | 33 | 2 | 40 |
| learned_v5 | 1 | 1 | 2 | 34 | 2 | 40 |
| prior | 0 | 1 | 0 | 37 | 2 | 40 |
