#!/usr/bin/env python
"""Reproduction script for all experiments in the paper.

This script reproduces all tables and figures from the paper.
Run each section separately as they require different compute resources.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_protocol() -> dict:
    """Load frozen protocol manifest for model snapshots and hashes."""
    path = PROJECT_ROOT / "data" / "manifests_frozen" / "protocol_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_model_snapshots() -> dict:
    protocol = load_protocol()
    return protocol["model_snapshots"]


def run(cmd: list[str], cwd: Path = None) -> None:
    """Run command and stream output."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def prepare_manifests():
    """Step 1: Prepare QASPER manifests from official data."""
    run([
        sys.executable, "scripts/prepare_qasper.py",
        "--qasper-dir", "data/qasper",
        "--out-dir", "data/manifests"
    ])


def freeze_protocol():
    """Step 2: Freeze protocol with paper-disjoint splits."""
    run([sys.executable, "scripts/freeze_protocol.py"])


def run_rollouts(split: str, question_limit: int, out_path: str):
    """Step 3: Run counterfactual rollouts for training."""
    model_snapshots = get_model_snapshots()
    run([
        sys.executable, "scripts/run_qasper_rollouts.py",
        "--split", split,
        "--manifest-dir", "data/manifests_frozen",
        "--out", out_path,
        "--config", "config.local.json",
        "--question-limit", str(question_limit),
        "--questions-per-paper", "3",
        "--resume",
        "--max-groups", "4",
        "--paper-batch-size", "16",
        "--model", model_snapshots["generator_openai"],
    ])


def train_policy(label: str, train_path: str, dev_path: str, out_path: str):
    """Step 4: Train parent/section gate checkpoint."""
    run([
        sys.executable, "scripts/train_tree_policy.py",
        "--train", train_path,
        "--dev", dev_path,
        "--out", out_path,
        "--label", label,
        "--estimator", "auto",
        "--selection-metric", "balanced_accuracy",
        "--seed", "42",
    ])


def run_dev_benchmark():
    """Step 5: Run full dev benchmark with all baselines/ablations."""
    model_snapshots = get_model_snapshots()
    run([
        sys.executable, "scripts/run_qasper_benchmark.py",
        "--manifest-dir", "data/manifests_frozen",
        "--paper-manifest", "qasper_dev_papers_frozen.jsonl",
        "--question-manifest", "qasper_dev_questions_frozen.jsonl",
        "--questions", "250",
        "--systems",
        "B0_bm25", "B1_dense", "B2_hybrid_rrf", "B3_flat_neural",
        "B4_static_hierarchy", "prior", "learned_v7",
        "learned_v7_parent_only", "learned_v7_section_only",
        "learned_v7_no_rollback", "learned_v7_no_verifier",
        "oracle_evidence", "oracle_context", "full_document",
        "--parent-checkpoint", "checkpoints/policy_parent_v7_final.joblib",
        "--section-checkpoint", "checkpoints/policy_section_v7_final.joblib",
        "--artifact-dir", "data/artifacts/v7_dev_full",
        "--report", "analysis/v7_dev_full.md",
        "--provider", "openai",
        "--model", model_snapshots["generator_openai"],
    ])


def run_main_test():
    """Step 6: Run main test on stratified unseen papers (once)."""
    model_snapshots = get_model_snapshots()
    run([
        sys.executable, "scripts/run_qasper_benchmark.py",
        "--manifest-dir", "data/manifests_test",
        "--paper-manifest", "qasper_test_stratified_papers.jsonl",
        "--question-manifest", "qasper_test_stratified_questions.jsonl",
        "--questions", "151",
        "--systems", "B3_flat_neural", "B4_static_hierarchy", "prior", "learned_v7",
        "--parent-checkpoint", "checkpoints/policy_parent_v7_final.joblib",
        "--section-checkpoint", "checkpoints/policy_section_v7_final.joblib",
        "--artifact-dir", "data/artifacts/v7_main_test",
        "--report", "analysis/v7_main_results.md",
        "--provider", "openai",
        "--model", model_snapshots["generator_openai"],
    ])


def run_generator_swap():
    """Step 7: Generator swap experiment (OpenAI vs Gemini)."""
    model_snapshots = get_model_snapshots()
    # Run with OpenAI
    run([
        sys.executable, "scripts/run_qasper_benchmark.py",
        "--manifest-dir", "data/manifests_test",
        "--paper-manifest", "qasper_test_stratified_papers.jsonl",
        "--question-manifest", "qasper_test_stratified_questions.jsonl",
        "--questions", "50",
        "--systems", "learned_v7",
        "--parent-checkpoint", "checkpoints/policy_parent_v7_final.joblib",
        "--section-checkpoint", "checkpoints/policy_section_v7_final.joblib",
        "--artifact-dir", "data/artifacts/generator_swap_openai",
        "--report", "analysis/generator_swap_openai.md",
        "--provider", "openai",
        "--model", model_snapshots["generator_openai"],
    ])

    # Run with Gemini
    run([
        sys.executable, "scripts/run_qasper_benchmark.py",
        "--manifest-dir", "data/manifests_test",
        "--paper-manifest", "qasper_test_stratified_papers.jsonl",
        "--question-manifest", "qasper_test_stratified_questions.jsonl",
        "--questions", "50",
        "--systems", "learned_v7",
        "--parent-checkpoint", "checkpoints/policy_parent_v7_final.joblib",
        "--section-checkpoint", "checkpoints/policy_section_v7_final.joblib",
        "--artifact-dir", "data/artifacts/generator_swap_gemini",
        "--report", "analysis/generator_swap_gemini.md",
        "--provider", "gemini",
        "--model", model_snapshots["generator_gemini"],
    ])


def run_scifact_ood():
    """Step 8: ODD evaluation on SciFact (rationale-attribution)."""
    model_snapshots = get_model_snapshots()
    run([
        sys.executable, "scripts/prepare_scifact.py",
        "--scifact-dir", "data/scifact",
        "--out-dir", "data/manifests"
    ])
    run([
        sys.executable, "scripts/run_qasper_benchmark.py",
        "--manifest-dir", "data/manifests",
        "--paper-manifest", "scifact_dev_papers.jsonl",
        "--question-manifest", "scifact_dev_questions.jsonl",
        "--questions", "100",
        "--dataset-name", "scifact-v1.0",
        "--split-name", "rationale-attribution-ood",
        "--systems", "learned_v7",
        "--parent-checkpoint", "checkpoints/policy_parent_v7_final.joblib",
        "--section-checkpoint", "checkpoints/policy_section_v7_final.joblib",
        "--artifact-dir", "data/artifacts/scifact_ood",
        "--report", "analysis/scifact_ood.md",
        "--provider", "openai",
        "--model", model_snapshots["generator_openai"],
    ])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=[
        "prepare", "freeze", "rollouts_train", "rollouts_dev",
        "train_parent", "train_section", "dev_benchmark",
        "main_test", "generator_swap", "scifact_ood", "all"
    ])
    args = parser.parse_args()

    if args.step in ("prepare", "all"):
        prepare_manifests()

    if args.step in ("freeze", "all"):
        freeze_protocol()

    if args.step in ("rollouts_train", "all"):
        run_rollouts("train", 800, "data/rollouts/rollouts_v7_train_frozen.jsonl")

    if args.step in ("rollouts_dev", "all"):
        run_rollouts("dev", 250, "data/rollouts/rollouts_v7_dev.jsonl")

    if args.step in ("train_parent", "all"):
        train_policy(
            "parent",
            "data/rollouts/rollouts_v7_train_frozen.jsonl",
            "data/rollouts/rollouts_v7_dev.jsonl",
            "checkpoints/policy_parent_v7_final.joblib"
        )

    if args.step in ("train_section", "all"):
        train_policy(
            "section",
            "data/rollouts/rollouts_v7_train_frozen.jsonl",
            "data/rollouts/rollouts_v7_dev.jsonl",
            "checkpoints/policy_section_v7_final.joblib"
        )

    if args.step in ("dev_benchmark", "all"):
        run_dev_benchmark()

    if args.step in ("main_test", "all"):
        run_main_test()

    if args.step in ("generator_swap", "all"):
        run_generator_swap()

    if args.step in ("scifact_ood", "all"):
        run_scifact_ood()


if __name__ == "__main__":
    main()
