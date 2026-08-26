# v4 Exploratory Failure Analysis (FROZEN)

Frozen: 2026-08-26. Do not reuse the numbers below as final results.
Protocol fixes applied AFTER this freeze: paper-scoped retrieval,
gold->leaf multi-segment mapping, leaf evidence selector (top-1 / margin /
sibling-delta), per-query artifact logging, paper-level splits.

## Setup
- QASPER-test unseen papers (30 papers, first 60 extractive questions), gpt-4o-mini @ temp=1.0 era rollouts; policy v4 checkpoint not yet involved in these systems' training data.
- Four systems sharing retrieval/rerank/generator/NLI.

## Raw table (frozen)
| metric | B_flat | B_static | prior | learned |
|---|---|---|---|---|
| citation_recall | .108 | .098 | .058 | .058 |
| evidence_span_recall | .135 | .167 | .098 | .082 |
| citation_precision | .032 | .025 | .013 | .013 |

Note: `.013` belongs to **prior and learned**, not B_static (.025).

## Correct interpretation
Flat does NOT dominate. Versus static expansion:
- citation_recall −9.3% relative
- evidence_span_recall **+23.7% relative**
- citation_precision −21.9% relative

=> Expansion rescues more gold spans but mis-attributes claims more often.
This is an evidence-completeness vs attribution-specificity trade-off, which is
exactly the phenomenon the Attribution-Risk-Aware method must gate.

## Candidate mechanisms (to be decomposed empirically next)
1. NLI best-of-many effect: more sibling candidates -> higher chance of a
   false-positive high score (AR drops while true precision drops too).
2. Over-citation: every child above threshold became a citation (no top-1).
3. Gold-label chunking mismatch: one gold paragraph split across children made
   single-best auto-labelling undercount gold leaves.
4. Genuine drift onto wrong siblings.

Protocol fixes (2)-(3) are now implemented; the decomposition run quantifies
each bucket before any reward v5 / retraining decision.

## Security note
API keys removed from config.local.json (now null); OPENAI_API_KEY moved to the
user environment. Rotate the previously committed Gemini key at Google AI
Studio before any repository share.
