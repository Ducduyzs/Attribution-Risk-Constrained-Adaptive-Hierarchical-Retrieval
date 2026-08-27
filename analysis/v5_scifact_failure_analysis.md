# Fresh V5 Failure Analysis

This is a diagnostic analysis of the frozen 40-paper test. It must not be used
to tune thresholds or select a replacement model.

Artifacts include generated and verified claim counts, so empty generation and verifier rejection are separated.

## Input provenance

- `data/artifacts/v5_scifact_ood/artifacts_B_flat.jsonl`: 40 rows, SHA-256 `8c4ca3c701532c2c42c283bec50911eca615dd31062fefe6fc26f9cc0613c605`
- `data/artifacts/v5_scifact_ood/artifacts_B_static.jsonl`: 40 rows, SHA-256 `5709c1cd02cb59b8d917b7345ee17f2ec7b5fe5cefaf7b40eab6398502efa198`
- `data/artifacts/v5_scifact_ood/artifacts_prior.jsonl`: 40 rows, SHA-256 `e17b2bfd8067c9707e66c3438ac0ce9f8f3f508d62bd1437638f55ce6c3e5cf0`
- `data/artifacts/v5_scifact_ood/artifacts_learned_v5.jsonl`: 40 rows, SHA-256 `152b9574e29df834fd69a922e5736ff9066c67fadfcf181411da5fca6d54c99c`

## Failure counts

| system | full_grounding_success | generation_empty | partial_grounding_success | total |
|---|---:|---:|---:|---:|
| B_flat | 14 | 22 | 4 | 40 |
| B_static | 23 | 6 | 11 | 40 |
| learned_v5 | 22 | 6 | 12 | 40 |
| prior | 22 | 9 | 9 | 40 |
