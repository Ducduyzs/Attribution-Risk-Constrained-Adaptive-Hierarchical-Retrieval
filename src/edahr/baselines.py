"""Baseline systems B0-B4 and the shared benchmark harness.

Ladder (all share ingestion/hierarchy/generation/verification unless noted):

  B0  BM25 lexical child retrieval, flat context, no rerank
  B1  Dense-only FAISS child retrieval, flat context, no rerank
  B2  BM25 + dense fused with Reciprocal Rank Fusion, no rerank
  B3  Full neural fusion (dense+sparse+ColBERT) + cross-encoder rerank,
      flat context -- no hierarchical adaptation
  B4  Static hierarchical merging: parents always win when enough children
      are retrieved (non-adaptive hierarchy)

The proposed system ("edahr") is the full adaptive pipeline.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .config import Settings
from .evaluation import (
    aggregate,
    answer_exact_match,
    answer_token_f1,
    aurc,
    bootstrap_ci,
    citation_f1,
    citation_precision,
    citation_recall,
    e_aurc,
    evidence_span_recall,
    hit_rate_at_k,
    latency_stats,
    ndcg_at_k,
    precision_at_k,
    provenance_accuracy,
    recall_at_k,
    reciprocal_rank,
    selective_accuracy_at_coverage,
)
from .hierarchy import HierarchyBuilder
from .interfaces import scoped_search
from .pipeline import AdaptiveHierarchicalPipeline, classify_query
from .policy import NeverMergePolicy, StaticMergePolicy, policies_from_settings
from .schemas import Hierarchy, Hit, ScientificDocument


# ---------------------------------------------------------------------------
# Retrievers used only by baselines
# ---------------------------------------------------------------------------

class Bm25ChildRetriever:
    """Dependency-free BM25 over child passages (Okapi BM25, k1/b tunable)."""

    def __init__(self, hierarchy: Hierarchy, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.hierarchy = hierarchy
        self.node_ids = list(hierarchy.child_ids)
        self.doc_tokens: list[list[str]] = [
            hierarchy.node(node_id).text.lower().split() for node_id in self.node_ids
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.document_frequency: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.document_frequency[term] += 1
        self.total_docs = len(self.node_ids)

    def _idf(self, term: str) -> float:
        frequency = self.document_frequency.get(term, 0)
        return math.log(1.0 + (self.total_docs - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, k: int, source: str | None = None) -> list[Hit]:
        query_terms = query.lower().split()
        scores: list[float] = []
        for index, tokens in enumerate(self.doc_tokens):
            length_norm = self.k1 * (
                1.0 - self.b + self.b * self.doc_lengths[index] / max(1.0, self.avgdl)
            )
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self._idf(term) * tf * (self.k1 + 1.0) / (tf + length_norm)
            scores.append(score)
        allowed = [
            row for row, node_id in enumerate(self.node_ids)
            if source is None or self.hierarchy.node(node_id).source == source
        ]
        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)[:k]
        return [
            Hit(node_id=self.node_ids[row], score=scores[row], rank=rank)
            for rank, row in enumerate(ranked, start=1)
        ]


class RrfRetriever:
    """Reciprocal-rank fusion of several retrievers."""

    def __init__(self, retrievers: Sequence, rrf_k: int = 60):
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.hierarchy = next(
            (
                getattr(retriever, "hierarchy", None)
                for retriever in self.retrievers
                if getattr(retriever, "hierarchy", None) is not None
            ),
            None,
        )

    def search(self, query: str, k: int, source: str | None = None) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)
        first_scores: dict[str, float] = {}
        for retriever in self.retrievers:
            hits = (
                scoped_search(retriever, self.hierarchy, query, max(k, 100), source)
                if self.hierarchy is not None
                else retriever.search(query, max(k, 100))
            )
            for hit in hits:
                fused[hit.node_id] += 1.0 / (self.rrf_k + hit.rank)
                first_scores.setdefault(hit.node_id, hit.score)
        ordered = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            Hit(
                node_id=node_id,
                score=fused[node_id],
                rank=rank,
                dense_score=first_scores.get(node_id, 0.0),
            )
            for rank, node_id in enumerate(ordered, start=1)
        ]


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_by_paper(
    document_ids: Sequence[str],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Paper-level train/calibration/test split (no leakage across splits)."""
    rng = random.Random(seed)
    ids = sorted(set(document_ids))
    rng.shuffle(ids)
    total = len(ids)
    train_end = round(total * ratios[0])
    calibration_end = train_end + round(total * ratios[1])
    return {
        "train": ids[:train_end],
        "calibration": ids[train_end:calibration_end],
        "test": ids[calibration_end:],
    }


def auto_label_gold_children(
    hierarchy: Hierarchy, record: dict, tau: float = 0.5
) -> tuple[set[str], list[str]]:
    """Resolve a record's gold evidence to child ids.

    Accepts explicit ``gold_child_ids`` and/or free-text ``gold_quotes``.
    Every child whose token-F1 with the quote reaches ``tau`` (or that contains
    the quote verbatim) is labelled gold, so a paragraph split across several
    chunks maps to all of them instead of an arbitrary best one -- otherwise
    citation precision is understated by construction.
    """
    gold_children = set(record.get("gold_child_ids") or ())
    allowed_sources = {str(key) for key in (record.get("gold_pages") or {})}
    if record.get("source"):
        allowed_sources.add(str(record["source"]))
    candidates = hierarchy.child_ids
    if allowed_sources:
        candidates = [
            child_id
            for child_id in hierarchy.child_ids
            if hierarchy.node(child_id).source in allowed_sources
        ]
    matched_quotes: list[str] = []
    for quote in record.get("gold_quotes") or ():
        normalized = " ".join(str(quote).lower().split())
        if not normalized:
            continue
        matches: list[str] = []
        for child_id in candidates:
            text = " ".join(hierarchy.node(child_id).text.lower().split())
            overlap = _quick_token_f1(normalized, text)
            if normalized in text:
                overlap = max(overlap, 1.0)
            if overlap >= tau:
                matches.append(child_id)
        if matches:
            gold_children.update(matches)
            matched_quotes.append(str(quote))
    return gold_children, matched_quotes or [str(q) for q in record.get("gold_quotes") or ()]


def _quick_token_f1(first: str, second: str) -> float:
    first_tokens, second_tokens = first.split(), second.split()
    if not first_tokens or not second_tokens:
        return 0.0
    second_counts = Counter(second_tokens)
    common = sum(
        min(count, second_counts.get(token, 0))
        for token, count in Counter(first_tokens).items()
    )
    if not common:
        return 0.0
    precision = common / len(first_tokens)
    recall = common / len(second_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

DEFAULT_CORRECTNESS = Callable[[float, float], float]


def default_correctness(answer_f1: float, citation_score: float) -> float:
    """A query is 'correct' when its answer overlaps gold AND cites gold evidence."""
    return float(answer_f1 >= 0.35 and citation_score > 0.0)


@dataclass
class BenchmarkRun:
    name: str
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_benchmark(
    name: str,
    pipeline: AdaptiveHierarchicalPipeline,
    records: Sequence[dict],
    ks: Sequence[int] = (3, 5, 10),
    correct_fn: DEFAULT_CORRECTNESS = default_correctness,
    seed: int = 42,
) -> BenchmarkRun:
    hierarchy = pipeline.hierarchy
    run = BenchmarkRun(name=name)
    for record in records:
        query = record["query"]
        gold_children, gold_quotes = auto_label_gold_children(hierarchy, record)
        gold_pages = {
            (str(source), int(page))
            for source, page in (record.get("gold_pages") or {}).items()
        }
        result = pipeline.answer(query, source=record.get("source"))
        ranked_ids = [hit.node_id for hit in result.hits]
        evidence_nodes = [evidence.node_id for evidence in result.evidence.values()]
        evidence_quotes = [evidence.quote for evidence in result.evidence.values()]
        provenance = [
            (evidence.source, evidence.page_start, evidence.page_end)
            for evidence in result.evidence.values()
        ]
        answer_text = " ".join(claim.text for claim in result.generation.claims).strip()
        mean_confidence = (
            sum(claim.confidence for claim in result.generation.claims)
            / len(result.generation.claims)
            if result.generation.claims
            else 0.0
        )
        evidence_node_set = {evidence.node_id for evidence in result.evidence.values()}
        retrieved_child_set = set(ranked_ids[: pipeline.settings.rerank_k])
        candidate_child_set = {
            child_id for block in result.context for child_id in block.evidence_ids
        }
        graded = {child_id: 1.0 for child_id in gold_children}
        citation_evaluable = bool(gold_children)
        row: dict = {
            "query": query,
            "question_id": str(record.get("question_id") or ""),
            "source": record.get("source"),
            "citation_evaluable": citation_evaluable,
        }
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, gold_children, k)
            row[f"precision@{k}"] = precision_at_k(ranked_ids, gold_children, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, graded, k)
            row[f"hit_rate@{k}"] = hit_rate_at_k(ranked_ids, gold_children, k)
        row["mrr"] = reciprocal_rank(ranked_ids, gold_children)
        row["evidence_span_recall"] = (
            evidence_span_recall(evidence_quotes, gold_quotes)
            if gold_quotes
            else 0.0
        )
        if citation_evaluable:
            row["citation_precision"] = citation_precision(evidence_node_set, gold_children)
            row["citation_recall"] = citation_recall(evidence_node_set, gold_children)
            row["citation_f1"] = citation_f1(evidence_node_set, gold_children)
        else:
            # Undefined, not zero: zero would silently depress macro grounding
            # metrics for questions whose gold paragraph could not be mapped.
            row["citation_precision"] = None
            row["citation_recall"] = None
            row["citation_f1"] = None
        row["provenance_accuracy"] = (
            provenance_accuracy(provenance, gold_pages) if gold_pages else 0.0
        )
        gold_answer = str(record.get("answer") or record.get("gold_answer") or "")
        row["answer_em"] = answer_exact_match(answer_text, gold_answer) if gold_answer else 0.0
        row["answer_f1"] = answer_token_f1(answer_text, gold_answer) if gold_answer else 0.0
        row["confidence"] = mean_confidence
        row["correct"] = (
            correct_fn(row["answer_f1"], float(row["citation_recall"]))
            if citation_evaluable else 0.0
        )
        row["context_tokens"] = float(result.metrics.get("context_tokens", 0.0))
        row["latency_ms"] = float(result.metrics.get("total_latency_ms", 0.0))
        # Per-query artifacts for failure decomposition and manual audit.
        row["gold_child_ids"] = sorted(gold_children)
        row["evidence_node_ids"] = sorted(evidence_node_set)
        row["retrieved_child_ids"] = sorted(retrieved_child_set)
        row["candidate_child_ids"] = sorted(candidate_child_set)
        row["rescued_leaf_ids"] = sorted(
            (evidence_node_set - retrieved_child_set) & gold_children
        )
        row["harmful_drift_leaf_ids"] = sorted(
            (evidence_node_set - retrieved_child_set) - gold_children
        )
        row["kept_correct_leaf_ids"] = sorted(
            (evidence_node_set & retrieved_child_set) & gold_children
        )
        row["kept_wrong_leaf_ids"] = sorted(
            (evidence_node_set & retrieved_child_set) - gold_children
        )
        row["claim_evidence"] = [
            {
                "claim": evidence.claim_text,
                "node_id": evidence.node_id,
                "support_score": evidence.support_score,
                "context_id": evidence.context_id,
            }
            for evidence in result.evidence.values()
        ]
        run.rows.append(row)

    accuracies = [float(row["correct"]) for row in run.rows]
    confidences = [float(row["confidence"]) for row in run.rows]
    latencies = [float(row["latency_ms"]) for row in run.rows]
    macro = aggregate(run.rows)
    citation_scores = [
        float(row["citation_f1"])
        for row in run.rows
        if isinstance(row.get("citation_f1"), (int, float))
    ]
    ci_low, ci_high = bootstrap_ci(citation_scores, seed=seed)
    run.summary = {
        **macro,
        **{f"latency_{key}": value for key, value in latency_stats(latencies).items()},
        "aurc": aurc(accuracies, confidences) if accuracies else 0.0,
        "e_aurc": e_aurc(accuracies, confidences) if accuracies else 0.0,
        "selective_accuracy@80cov": (
            selective_accuracy_at_coverage(accuracies, confidences, 0.8)
            if accuracies else 0.0
        ),
        "citation_f1_ci_low": ci_low,
        "citation_f1_ci_high": ci_high,
        "num_queries": float(len(run.rows)),
        "citation_evaluable_queries": float(len(citation_scores)),
    }
    return run


def significance_vs_baseline(
    proposed: BenchmarkRun, baseline: BenchmarkRun, metric: str = "citation_f1", seed: int = 42
) -> float:
    from .evaluation import paired_bootstrap_test

    paired = _paired_metric_rows(proposed, baseline, metric)
    firsts, seconds = zip(*paired) if paired else ((0.0,), (0.0,))
    return paired_bootstrap_test(list(firsts), list(seconds), seed=seed)


def clustered_ci_vs_baseline(
    proposed: BenchmarkRun,
    baseline: BenchmarkRun,
    metric: str = "citation_f1",
    seed: int = 42,
) -> tuple[float, float]:
    """Paper-clustered CI for the paired proposed-minus-baseline difference."""
    from .evaluation import paired_cluster_bootstrap

    baseline_by_key = {_row_identity(row): row for row in baseline.rows}
    diffs: list[float] = []
    clusters: list[str] = []
    for row in proposed.rows:
        other = baseline_by_key.get(_row_identity(row))
        if other is None:
            continue
        first, second = row.get(metric), other.get(metric)
        if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
            continue
        diffs.append(float(first) - float(second))
        clusters.append(str(row.get("source") or "unknown-paper"))
    return paired_cluster_bootstrap(diffs, clusters, seed=seed)


def _row_identity(row: dict) -> tuple[str, ...]:
    question_id = str(row.get("question_id") or "")
    source = str(row.get("source") or "")
    if question_id:
        return source, question_id
    return source, str(row.get("query") or "")


def _paired_metric_rows(
    proposed: BenchmarkRun, baseline: BenchmarkRun, metric: str
) -> list[tuple[float, float]]:
    baseline_by_key = {_row_identity(row): row for row in baseline.rows}
    pairs: list[tuple[float, float]] = []
    for row in proposed.rows:
        other = baseline_by_key.get(_row_identity(row))
        if other is None:
            continue
        first, second = row.get(metric), other.get(metric)
        if isinstance(first, (int, float)) and isinstance(second, (int, float)):
            pairs.append((float(first), float(second)))
    return pairs


# ---------------------------------------------------------------------------
# Baseline construction from shared heavy components
# ---------------------------------------------------------------------------

def build_documents(documents: list[ScientificDocument], settings: Settings) -> Hierarchy:
    return HierarchyBuilder(settings).build(documents)


def make_baseline_pipeline(
    name: str,
    hierarchy: Hierarchy,
    *,
    encoder=None,
    index_factory=None,
    reranker=None,
    generator=None,
    verifier=None,
    settings: Settings | None = None,
) -> AdaptiveHierarchicalPipeline:
    """Wire one of B0..B4 / 'edahr' from shared heavy components.

    ``index_factory(settings)`` must return a configured retrievable index
    (typically :class:`edahr.index.MultiRepresentationIndex`); BM25 is built
    locally without extra dependencies.
    """
    settings = settings or Settings()

    if name == "B0_bm25":
        variant = replace(settings, use_dense=False, use_sparse=False, use_colbert=False)
        pipeline_settings = replace(variant)
        retriever = Bm25ChildRetriever(hierarchy, settings.bm25_k1, settings.bm25_b)
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=retriever, reranker=reranker,
            generator=generator, verifier=verifier, settings=pipeline_settings,
            policy=NeverMergePolicy(), rerank_enabled=False,
        )
    if name == "B1_dense":
        variant = replace(settings, use_dense=True, use_sparse=False, use_colbert=False)
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=index_factory(variant), reranker=reranker,
            generator=generator, verifier=verifier, settings=variant,
            policy=NeverMergePolicy(), rerank_enabled=False,
        )
    if name == "B2_hybrid_rrf":
        variant = replace(
            settings,
            use_dense=True, use_sparse=False, use_colbert=False,
            fusion_mode="rrf",
        )
        bm25 = Bm25ChildRetriever(hierarchy, settings.bm25_k1, settings.bm25_b)
        dense_retriever = index_factory(variant)
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy,
            retriever=RrfRetriever([bm25, dense_retriever], settings.rrf_k),
            reranker=reranker,
            generator=generator,
            verifier=verifier,
            settings=replace(variant, expansion_max_depth=0),
            policy=NeverMergePolicy(),
            rerank_enabled=False,
        )
    if name == "B3_flat_neural":
        variant = replace(settings, expansion_max_depth=0)
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=index_factory(variant), reranker=reranker,
            generator=generator, verifier=verifier, settings=variant,
            policy=NeverMergePolicy(), rerank_enabled=True,
        )
    if name == "B4_static_hierarchy":
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=index_factory(replace(settings)),
            reranker=reranker, generator=generator, verifier=verifier,
            settings=settings, policy=StaticMergePolicy(), rerank_enabled=True,
        )
    if name == "edahr":
        parent_policy, section_policy = policies_from_settings(settings)
        return AdaptiveHierarchicalPipeline(
            hierarchy=hierarchy, retriever=index_factory(replace(settings)),
            reranker=reranker, generator=generator, verifier=verifier,
            settings=settings,
            parent_policy=parent_policy,
            section_policy=section_policy,
            rerank_enabled=True,
        )
    raise ValueError(f"Unknown baseline: {name}")


BASELINE_NAMES: tuple[str, ...] = (
    "B0_bm25", "B1_dense", "B2_hybrid_rrf", "B3_flat_neural",
    "B4_static_hierarchy", "edahr",
)
