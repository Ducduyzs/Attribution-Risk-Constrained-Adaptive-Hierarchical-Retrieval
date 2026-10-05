# Do Scientific QA Systems Need Retrieval Trees? A Preregistered Study of Evidence Selection under Context Budgets

*Draft v1 — 2026-10-05. Venue-agnostic; numbers trace to `analysis/*.md` in the repository.
Items marked [CHECK] need verification before submission.*

## Abstract

Hierarchical retrieval methods such as RAPTOR build summary trees over documents so that a
reader can draw on context at several granularities. For question answering over a single
scientific paper with verifiable citations, we ask whether such structure is worth its
indexing cost. We start from an attribution-risk-aware adaptive expansion system and report
what a controlled evaluation reveals: (i) expanding retrieved passages to their parents or
sections adds unretrieved gold evidence for only about 1% of candidate groups on QASPER, so
learned expansion gates fail to generalize (dev AUC 0.51–0.54); (ii) giving the reader the
whole paper does not beat flat retrieval, while gold-evidence context does (+0.23 evidence
F1), placing the bottleneck in evidence selection; (iii) with a fixed token budget, gold
evidence is in the candidate pool for 94–99% of questions but only ~45% of gold passages
reach a 1,024-token context. Tracing RAPTOR's apparent advantage to its query–passage
similarity signal, we propose Agreement Ranking, a parameter-free fusion of a cross-encoder
with SBERT similarity that needs no tree or summaries. In preregistered evaluations on
unseen QASPER papers (100 dev, 180 test) and on PeerQA, Agreement Ranking is non-inferior to
RAPTOR in official evidence F1 with GPT-4o-mini (test: +0.020, Holm p = 0.006) at roughly
1/400 of the indexing compute; plain cross-encoder reranking of all passages is likewise
non-inferior on PeerQA. With an open 7B reader the QASPER non-inferiority is not
established, and no selector is shown to beat plain reranking. We also document a verifier
defect that silently rejected about a quarter of correct claims across all systems.

## 1 Introduction

Retrieval-augmented question answering over long scientific documents must decide *what*
text the reader sees under a context budget and must attribute each claim to evidence.
Hierarchical approaches (RAPTOR; HiChunk; Mix-of-Granularity) argue that multi-granularity
context helps. They are, however, expensive: RAPTOR clusters passages and summarizes each
cluster with a language model before any question is asked.

We set out to improve such systems with an *attribution-risk-constrained* adaptive expansion
policy that decides, per query, whether to expand retrieved passages to their parents or
sections. A controlled, preregistered evaluation instead showed that, on single-paper
scientific QA, the structural premise behind expansion has little headroom, and that the
advantage of a strong hierarchical baseline can be matched by a much cheaper signal.

**Contributions.**
1. A diagnostic decomposition of where gold evidence is lost (candidate pool → packed context
   → generated citations → verified citations) showing that packing under a budget, not
   retrieval recall, is the main loss (§4).
2. Quantified negative results: expansion headroom (~1%), learned expansion gates, attribution
   drift control (conformal risk control), contextual chunk embeddings, and full-document
   reading all fail to help in this setting (§4, §7).
3. Agreement Ranking, a tree-free selector derived from a mechanism analysis of RAPTOR, which
   is non-inferior to RAPTOR under preregistered tests on unseen data at ~1/100–1/400 of its
   indexing cost (§5–6), with explicit robustness limits across readers and datasets.
4. Methodological lessons: a verifier safety guard that vetoed ~25% of entailed claims, and a
   generator-provenance error, both of which would have changed published conclusions (§3.4).

## 2 Related Work

*Hierarchical and adaptive retrieval.* RAPTOR (Sarthi et al., 2024) builds recursive
cluster-summary trees; LongRAG (Jiang et al., 2024) retrieves long units; HiChunk (Lu et al.,
ACL 2026) and Mix-of-Granularity (Zhong et al., COLING 2025) adapt chunk granularity to the
query; Dense X Retrieval (Chen et al., EMNLP 2024) argues for proposition units. Adaptive-RAG
(Jeong et al., NAACL 2024), Self-RAG (Asai et al., ICLR 2024) and CRAG decide when and how
much to retrieve.

*Context selection and packing.* FILCO (Wang et al., 2023), RECOMP (Xu et al., ICLR 2024),
Lost in the Middle (Liu et al., TACL 2024) and Sufficient Context (Joren et al., ICLR 2025)
study what reaches the reader. [CHECK: add AdaGReS and "Recall Is Not Enough" — cited by the
v9 design notes; verify bibliographic details.]

*Attribution.* ALCE (Gao et al., EMNLP 2023), Attribute or Abstain (Buchmann et al., EMNLP
2024), LongCite (Zhang et al., Findings ACL 2025), AttributionBench (Li et al., Findings ACL
2024), MiniCheck (Tang et al., EMNLP 2024). *Risk control*: C-RAG (Kang et al., ICML 2024),
conformal factuality (Mohri and Hashimoto, ICML 2024), conformal risk control (Angelopoulos et
al., ICLR 2024). *Scientific QA*: QASPER (Dasigi et al., NAACL 2021), PeerQA (Baumgärtner et
al., NAACL 2025), SciDQA (Singh et al., EMNLP 2024).

## 3 Experimental Framework

### 3.1 Pipeline
Papers are split into sections, parents (4 leaves, overlap 1) and ~220-token leaves.
Candidates are retrieved with BGE-M3 (dense+sparse+multi-vector) and reranked with
bge-reranker-v2-m3. A context packer (deduplication, top-k, 0/1 knapsack) fills an exact
cl100k token budget including source headers. The reader emits atomic claims citing context
IDs under a strict JSON schema; a DeBERTa-v3 NLI verifier keeps a claim only if some cited
leaf entails it, reading only the text actually shown to the reader.

### 3.2 Data
**QASPER v0.3** with the official AllenAI evaluator vendored and verified question-by-question
(30,690 synthetic predictions, 0 mismatches after fixing three converter/metric divergences).
Splits: a 75-question / 30-paper exploratory dev set (used for all design work), a fresh
100-question / 100-paper confirmation set from dev papers never used before, and a 180-question
stratified test set opened once. **PeerQA** via its MTEB packaging (NLPeer papers, 70 papers,
136 questions, author-annotated evidence at sentence-like unit level; median ~5,200 words).

### 3.3 Metrics and statistics
Primary: official paragraph evidence F1 (max over annotators); also leaf-level citation F1,
answer token F1, packed gold-character coverage, context and API tokens. Papers are the
sampling unit: paired cluster sign-flip tests, paper bootstrap CIs, Cohen's d_z, Holm
correction over each preregistered family, non-inferiority at an absolute margin of 0.02.

### 3.4 Two defects found during the audit
*Verifier guard.* A lexical polarity/number check meant to gate a lexical fallback was applied
to the whole 220-token leaf and vetoed NLI-entailed claims whenever the leaf contained any
negation or unrelated number: ~25% of claims were rejected, ~60% of them on gold leaves.
Fixing it raised citation F1 by 0.07–0.11 for every system (paper-clustered CIs exclude 0 in
13/15 system×metric cells) without changing system rankings. *Provenance.* An earlier results
table recorded `gpt-4o-mini` as the reader while a hard-coded local Qwen2.5-7B produced it.
All results below use the fixed verifier and verified reader identity.

## 4 Where Does Hierarchical Expansion Help? Diagnostics

**Expansion headroom.** Over 2,881 train and 924 dev candidate groups, expanding to the parent
adds a gold leaf that retrieval had not already returned in 28 and 8 groups (~1%); section
expansion beyond the parent in ~1.2–1.3%. Harmful attribution drift from expansion is likewise
negligible (0–1 leaves per 75 questions), so drift-control mechanisms (citable-leaf
restriction; conformal thresholds) had nothing to control.

**Learned gates.** Gates trained on counterfactual KEEP/EXPAND rollouts reach train AUC 0.995
but dev AUC 0.537 (parent) and 0.511 (section); paper-grouped CV ≈ 0.50 for RF/GB/HGB.
Pairwise learning and query-relative features raise dev AUC to ~0.585, below a preregistered
0.60 bar; a gold-derived target looks learnable (AUC 0.9) only through a structural leak
("does the node contain unretrieved leaves"), and is at chance conditionally.

**Reader-side ceilings (75 dev questions, GPT-4o-mini).**

| System | Citation F1 | Evidence F1 | Answer F1 | Context tokens |
|---|---:|---:|---:|---:|
| Gold evidence only (oracle) | 0.621 | 0.602 | 0.270 | 517 |
| RAPTOR | 0.436 | 0.404 | 0.244 | 1,452 |
| Adaptive expansion (prior) | 0.387 | 0.396 | 0.231 | 2,215 |
| Static hierarchy | 0.365 | 0.373 | 0.229 | 2,932 |
| Flat retrieval | 0.370 | 0.369 | 0.201 | 1,572 |
| Full document | 0.343 | 0.337 | 0.205 | 4,355 |
| LongRAG | 0.313 | 0.287 | 0.195 | 1,562 |

Reading the whole paper does not beat flat retrieval (evidence F1 −0.032 [−0.088, +0.024]) at
2.8× the tokens; gold-only context is far better (+0.233 [+0.164, +0.296]).

**Packing is the loss point.** Under exact budgets (512/1,024/2,048 tokens) gold evidence is in
the reranked pool for 94–99% of questions, yet only ~45% of gold leaves reach a 1,024-token
context. Reranking all leaves rather than the top 24 barely changes this.

## 5 Agreement Ranking

**Mechanism.** All budgeted arms share one packer; RAPTOR differs only in its candidate set
(12.8 vs 29.3 leaves). Its pool removes about one of the 2.7 non-gold leaves the cross-encoder
ranks above the best gold leaf (best-gold rank 3.73 → 2.72). Testing what supplies this
signal on cached rankings: neighbour-support smoothing and summary-free clustering do not
reproduce it; fusing the cross-encoder with SBERT query–leaf similarity (RAPTOR's own encoder,
multi-qa-mpnet-base-cos-v1) does.

**Method.** Score every leaf with the cross-encoder; compute SBERT cosine between query and
leaf; order leaves by reciprocal-rank fusion (k = 60); assign the fused order the
cross-encoder's own score multiset so that the downstream packer is unchanged. No training,
no tree, no summarizer.

## 6 Preregistered Results

Primary metric: official evidence F1 averaged over the three budgets per question.
Non-inferiority margin 0.02; Holm over each family.

| Setting | Agreement vs RAPTOR (NI) | All-leaf vs RAPTOR (NI) | Agreement vs all-leaf (sup.) |
|---|---|---|---|
| QASPER confirm, 100 unseen papers, GPT-4o-mini | +0.032 [−0.006, +0.072], Holm p = 0.010 ✔ | — | +0.023, p = 0.149 ✘ |
| QASPER test, 180, GPT-4o-mini | +0.020 [−0.007, +0.048], Holm p = 0.006 ✔ | −0.008, p = 0.201 ✘ | +0.028, Holm p = 0.052 ✘ |
| PeerQA, 136, GPT-4o-mini | +0.011 [−0.003, +0.025], Holm p < 0.001 ✔ | +0.005, Holm p < 0.001 ✔ | +0.006, p = 0.22 ✘ |
| QASPER confirm, Qwen2.5-7B | −0.033 [−0.074, +0.008] ✘ | — | −0.043 ✘ |
| QASPER test, Qwen2.5-7B | +0.007 [−0.027, +0.043], Holm p = 0.19 ✘ | −0.007 ✘ | +0.014 ✘ |
| PeerQA, Qwen2.5-7B | +0.005 [−0.016, +0.024], Holm p = 0.016 ✔ | +0.002, Holm p = 0.015 ✔ | +0.003 ✘ |

**Per budget (QASPER test, GPT-4o-mini), evidence F1:** RAPTOR 0.346 / 0.353 / 0.372;
all-leaf 0.310 / 0.361 / 0.378; Agreement 0.357 / 0.384 / 0.390.

**Indexing cost (RTX 3090):** RAPTOR 7,044 s (100 papers), 12,151 s (180), 3,097 s (70 PeerQA);
Agreement 17 s, 29 s, 32 s.

**Findings.** (1) With GPT-4o-mini, Agreement Ranking is non-inferior to RAPTOR on all three
unseen evaluations. (2) On PeerQA's longer papers, plain reranking is also non-inferior, and
RAPTOR packs the least gold evidence at every budget. (3) With Qwen2.5-7B the QASPER
non-inferiority is not established: given identical contexts, reader-specific citation
behaviour dominates the small differences between selectors. (4) No selector is shown to beat
plain cross-encoder reranking of all passages. (5) A dev-set observation that RAPTOR packs
more gold than all-leaf reranking did not replicate on unseen papers.

## 7 Further Negative Results (exploratory dev)

- Contextual chunk embeddings (title prefix; Anthropic-style LLM contexts for 870 leaves,
  $0.43) leave BGE-M3 systems unchanged and hurt RAPTOR (−0.06 to −0.08 citation F1): with
  ~26 leaves per paper and candidate_k = 80, the raw-text reranker decides the ranking.
- Expanding on top of RAPTOR retrieval loses 0.038 citation F1 and adds 29% tokens.
- Allowing 2–3 citations per claim adds only +0.01–0.03 citation F1; even with gold-only
  context citation recall stays ≈ 0.55 because the reader emits ~2 claims.
- Heuristic dependency-preserving evidence groups, sentence windows, coverage selection and
  evidence-first generation do not beat RAPTOR; dependency grouping is the weakest arm.

## 8 Limitations

One reader family per main result; QASPER and a 70-paper PeerQA subset only; automatic
evidence metrics (a blinded human audit sample exists but is unlabelled); the absolute 0.02
margin is relatively loose on PeerQA's sentence-level F1 scale; RAPTOR uses a BART summarizer
substitution rather than the original GPT-3.5 summaries; the 75-question dev set was reused
for many exploratory analyses and is not used for any confirmatory claim.

## 9 Conclusion

For single-paper scientific QA with verifiable citations, building retrieval trees did not
pay for itself: the information they add for evidence selection can be matched by a cheap
query–passage similarity signal, and on longer papers by plain reranking. The remaining gap
to gold-evidence context lies in selecting and packing evidence under a budget and in how
readers cite it — not in hierarchical structure.

## Reproducibility

Code, frozen manifests (with SHA-256), protocols and per-question artifacts:
`analysis/v10_selector_design.md`, `v10_test_protocol.md`, `v11_protocol.md`; runners
`scripts/run_v10_confirm.py`, `scripts/run_v11.py`; official evaluator audit
`scripts/audit_evaluator.py`.

## References [CHECK all bibliographic details]

See `analysis/literature_review_improvement_directions.md` (40 entries; ✔ verified, ○ to verify).
