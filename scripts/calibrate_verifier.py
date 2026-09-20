"""Calibrate verifier on dev set and report false acceptance rates."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edahr.config import Settings
from edahr.pipeline import AdaptiveHierarchicalPipeline
from edahr.qasper import documents_from_paper_records, read_jsonl
from edahr.runtime import build_pipeline_from_documents
from edahr.verification import verify_generation


def load_dev_data():
    manifest_dir = PROJECT_ROOT / "data" / "manifests_frozen"
    papers = list(read_jsonl(manifest_dir / "qasper_dev_papers_frozen.jsonl"))
    questions = list(read_jsonl(manifest_dir / "qasper_dev_questions_frozen.jsonl"))
    return papers, questions


def main():
    papers, questions = load_dev_data()
    
    config_path = PROJECT_ROOT / "config.local.json"
    settings = Settings.from_json(config_path)
    
    documents = documents_from_paper_records(papers)
    pipeline = build_pipeline_from_documents(documents, settings)
    
    print(f"Loaded {len(papers)} papers, {len(questions)} questions")
    print(f"Verifier config:")
    print(f"  nli_support_threshold: {settings.nli_support_threshold}")
    print(f"  nli_contradiction_threshold: {settings.nli_contradiction_threshold}")
    print(f"  claim_confidence_threshold: {settings.claim_confidence_threshold}")
    print(f"  lexical_support_min_coverage: {settings.lexical_support_min_coverage}")
    print(f"  sibling_threshold_delta: {settings.sibling_threshold_delta}")
    print(f"  evidence_margin: {settings.evidence_margin}")
    print(f"  max_evidence_per_claim: {settings.max_evidence_per_claim}")
    
    # Run verification on a sample to check false acceptance
    sample_questions = questions[:50]
    
    stats = {
        "total_claims": 0,
        "verified_claims": 0,
        "rejected_claims": 0,
        "lexical_fallback_claims": 0,
        "contradiction_detected": 0,
        "unsupported_claims": 0,
    }
    
    for q in sample_questions:
        result = pipeline.answer(q["query"], source=q.get("source"))
        
        for claim in result.generation.claims:
            stats["total_claims"] += 1
            if claim.support_score >= settings.nli_support_threshold:
                stats["verified_claims"] += 1
            else:
                stats["rejected_claims"] += 1
                # Check if it was a contradiction
                # (This would require deeper inspection of verification trace)
        
        # Check verification trace
        for trace in result.verification_trace:
            if trace.get("decision") == "contradiction":
                stats["contradiction_detected"] += 1
            if trace.get("decision") == "lexical_fallback":
                stats["lexical_fallback_claims"] += 1
            if trace.get("decision") == "reject":
                stats["unsupported_claims"] += 1
    
    print("\nVerifier statistics on dev sample (50 questions):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    if stats["total_claims"] > 0:
        print(f"\n  Verification rate: {stats['verified_claims']/stats['total_claims']:.2%}")
        print(f"  Rejection rate: {stats['rejected_claims']/stats['total_claims']:.2%}")
        print(f"  Contradiction rate: {stats['contradiction_detected']/stats['total_claims']:.2%}")
        print(f"  Lexical fallback rate: {stats['lexical_fallback_claims']/stats['total_claims']:.2%}")


if __name__ == "__main__":
    main()