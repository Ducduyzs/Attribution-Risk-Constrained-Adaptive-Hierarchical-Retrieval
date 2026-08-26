from __future__ import annotations

from .config import Settings
from .interfaces import Verifier
from .text import claim_coverage
from .schemas import (
    Claim,
    ContextBlock,
    Evidence,
    Generation,
    Hierarchy,
    Level,
)


def _candidate_children(
    cited_blocks: list[ContextBlock], hierarchy: Hierarchy
) -> list[str]:
    """Children actually reachable from the cited context, in reading order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for block in cited_blocks:
        for evidence_id in block.evidence_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            node = hierarchy.nodes.get(evidence_id)
            if node is not None and node.level == Level.CHILD:
                ordered.append(evidence_id)
    return ordered


def verify_generation(
    generation: Generation,
    context: list[ContextBlock],
    hierarchy: Hierarchy,
    verifier: Verifier,
    settings: Settings,
    claim_supports: list[tuple[str, float]] | None = None,
    retrieved_ids: set[str] | None = None,
) -> tuple[Generation, dict[str, Evidence], dict[str, float]]:
    """Child-level NLI verification with leaf-level attribution gating.

    For every generated claim we run entailment against each descendant child
    of the cited context -- never against the whole parent/context blob.
    Selection then enforces precision on top of recall:

    * siblings outside the initially retrieved set must clear
      ``nli_support_threshold + sibling_threshold_delta``;
    * at most ``max_evidence_per_claim`` children become citations (top-1 by
      default);
    * when ``top1 - top2 < evidence_margin`` the choice is ambiguous, so only
      the top child is kept and the claim is flagged in metrics.

    When ``claim_supports`` is a list, ``(claim_text, best_support)`` pairs are
    appended for every claim reaching the NLI stage (pre-threshold best score).
    """
    context_by_id = {block.context_id: block for block in context}
    accepted_claims: list[Claim] = []
    evidence: dict[str, Evidence] = {}
    support_values: list[float] = []
    nli_calls = 0
    rejected_no_child = 0
    ambiguous_claims = 0
    sibling_filtered = 0

    def add_evidence(claim: Claim, child_id: str, score: float, block: ContextBlock) -> str:
        node = hierarchy.node(child_id)
        evidence_id = f"E{len(evidence) + 1}"
        evidence[evidence_id] = Evidence(
            evidence_id=evidence_id,
            node_id=node.node_id,
            source=node.source,
            page_start=node.page_start,
            page_end=node.page_end,
            quote=node.text,
            support_score=round(float(score), 4),
            char_start=node.char_start,
            char_end=node.char_end,
            claim_text=claim.text,
            context_id=block.context_id,
            confidence=node.confidence,
        )
        return evidence_id

    for claim in generation.claims:
        cited = [context_by_id[cid] for cid in claim.citations if cid in context_by_id]
        if not cited or claim.confidence < settings.claim_confidence_threshold:
            continue
        candidates = _candidate_children(cited, hierarchy)[: settings.max_children_per_claim]
        scored: list[tuple[str, float, ContextBlock]] = []
        best_support = 0.0
        for child_id in candidates:
            child_text = hierarchy.node(child_id).text
            score = verifier.support_score(claim.text, child_text)
            nli_calls += 1
            best_support = max(best_support, float(score))
            if score < settings.nli_support_threshold:
                # Deterministic fallback: near-verbatim restatements (including
                # numeral paraphrases like "six" vs "N = 6") that the NLI
                # checkpoint scores as neutral still count as supported.
                coverage = claim_coverage(claim.text, child_text)
                if (
                    settings.lexical_support_min_coverage > 0.0
                    and coverage >= settings.lexical_support_min_coverage
                ):
                    score = max(score, coverage)
            if score >= settings.nli_support_threshold:
                origin = next(block for block in cited if child_id in block.evidence_ids)
                scored.append((child_id, float(score), origin))
        if not scored:
            rejected_no_child += 1
            if claim_supports is not None:
                claim_supports.append((claim.text, round(best_support, 4)))
            continue
        if claim_supports is not None:
            claim_supports.append((claim.text, round(best_support, 4)))
        scored.sort(key=lambda item: item[1], reverse=True)
        if retrieved_ids and settings.sibling_threshold_delta > 0.0:
            bar = settings.nli_support_threshold + settings.sibling_threshold_delta
            before = len(scored)
            scored = [
                entry for entry in scored
                if entry[0] in retrieved_ids or entry[1] >= bar
            ]
            sibling_filtered += before - len(scored)
        if not scored:
            rejected_no_child += 1
            continue
        ambiguous = (
            len(scored) > 1
            and (scored[0][1] - scored[1][1]) < settings.evidence_margin
        )
        limit = settings.max_evidence_per_claim
        kept = scored[:limit] if limit and limit > 0 else scored
        if ambiguous:
            kept = kept[:1]
            ambiguous_claims += 1
        support_values.extend(score for _, score, _ in kept)
        verified_citations = tuple(
            add_evidence(claim, child_id, score, block)
            for child_id, score, block in kept
        )
        accepted_claims.append(
            Claim(text=claim.text, citations=verified_citations, confidence=claim.confidence)
        )

    verified = Generation(
        answerable=generation.answerable and bool(accepted_claims),
        claims=tuple(accepted_claims),
        reason=(
            generation.reason if accepted_claims
            else "No generated claim had a child passage pass NLI verification."
        ),
    )
    metrics = {
        "generated_claims": float(len(generation.claims)),
        "verified_claims": float(len(accepted_claims)),
        "claim_precision_proxy": len(accepted_claims) / max(1, len(generation.claims)),
        "mean_nli_support": sum(support_values) / max(1, len(support_values)),
        "child_nli_calls": float(nli_calls),
        "claims_rejected_no_child_support": float(rejected_no_child),
        "verified_evidence_children": float(len(evidence)),
        "ambiguous_claims": float(ambiguous_claims),
        "sibling_filtered_children": float(sibling_filtered),
    }
    return verified, evidence, metrics
