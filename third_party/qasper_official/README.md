# Official QASPER evaluator (vendored, unmodified)

- Source: https://github.com/allenai/qasper-led-baseline/blob/e996b6c7b1b5f95d9308a74e3586416c6e780df1/scripts/evaluator.py
- Upstream commit: `e996b6c7b1b5f95d9308a74e3586416c6e780df1` (2021-05-20)
- License: Apache-2.0 (`LICENSE`, copied from the same commit)
- `evaluator.py` SHA-256: `781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe`
- Fetched: 2026-09-29

Do not edit. `scripts/audit_evaluator.py` refuses to run if the hash changes,
and uses this file to check the local metrics (`edahr.evaluation`) and
converter (`edahr.qasper.convert_qasper`) question-by-question on the raw
QASPER v0.3 dev and test JSON.
