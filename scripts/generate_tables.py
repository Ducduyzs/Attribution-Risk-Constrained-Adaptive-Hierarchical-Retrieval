#!/usr/bin/env python
"""Generate all tables and figures from per-query artifacts."""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_artifacts(artifact_dir: Path) -> pd.DataFrame:
    """Load all artifact JSONL files into a single DataFrame."""
    dfs = []
    for path in artifact_dir.glob("artifacts_*.jsonl"):
        system = path.stem.replace("artifacts_", "")
        df = pd.read_json(path, lines=True)
        df["system"] = system
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def generate_main_table(artifact_dir: Path, output_path: Path):
    """Generate main results table (Table 1 in paper)."""
    df = load_artifacts(artifact_dir)
    if df.empty:
        print(f"No artifacts found in {artifact_dir}")
        return

    # Aggregate by system
    metrics = [
        "official_qasper_evidence_f1",
        "citation_f1",
        "answer_f1",
        "evidence_span_recall",
        "context_tokens",
        "latency_ms",
    ]
    
    summary = df.groupby("system")[metrics].mean().round(4)
    summary.to_csv(output_path.with_suffix(".csv"))
    
    # Markdown table
    md = "| System | Evidence F1 | Citation F1 | Answer F1 | Span Recall | Context Tokens | Latency (ms) |\n"
    md += "|---|---:|---:|---:|---:|---:|---:|\n"
    for sys_name, row in summary.iterrows():
        md += f"| {sys_name} | {row['official_qasper_evidence_f1']:.4f} | "
        md += f"{row['citation_f1']:.4f} | {row['answer_f1']:.4f} | "
        md += f"{row['evidence_span_recall']:.4f} | "
        md += f"{row['context_tokens']:.1f} | {row['latency_ms']:.1f} |\n"
    
    output_path.write_text(md)
    print(f"Generated {output_path}")


def generate_ablation_table(artifact_dir: Path, output_path: Path):
    """Generate ablation study table."""
    df = load_artifacts(artifact_dir)
    if df.empty:
        return

    ablation_systems = [
        "learned_v7",
        "learned_v7_parent_only",
        "learned_v7_section_only",
        "learned_v7_no_rollback",
        "learned_v7_no_verifier",
    ]
    
    df_ablation = df[df["system"].isin(ablation_systems)]
    if df_ablation.empty:
        return

    metrics = [
        "official_qasper_evidence_f1",
        "citation_f1",
        "answer_f1",
        "evidence_span_recall",
        "context_tokens",
        "latency_ms",
    ]
    
    summary = df_ablation.groupby("system")[metrics].mean().round(4)
    summary.to_csv(output_path.with_suffix(".csv"))
    
    md = "| System | Evidence F1 | Citation F1 | Answer F1 | Span Recall | Context Tokens | Latency (ms) |\n"
    md += "|---|---:|---:|---:|---:|---:|---:|\n"
    for sys_name, row in summary.iterrows():
        md += f"| {sys_name} | {row['official_qasper_evidence_f1']:.4f} | "
        md += f"{row['citation_f1']:.4f} | {row['answer_f1']:.4f} | "
        md += f"{row['evidence_span_recall']:.4f} | "
        md += f"{row['context_tokens']:.1f} | {row['latency_ms']:.1f} |\n"
    
    output_path.write_text(md)
    print(f"Generated {output_path}")


def generate_rescue_harm_analysis(artifact_dir: Path, output_path: Path):
    """Generate rescue vs harmful drift analysis (Table 2 in paper)."""
    df = load_artifacts(artifact_dir)
    if df.empty:
        return

    # Compute rescue and harmful drift rates
    results = []
    for sys_name, group in df.groupby("system"):
        total_rescued = group["rescued_leaf_ids"].apply(len).sum()
        total_harmful = group["harmful_drift_leaf_ids"].apply(len).sum()
        total_kept_correct = group["kept_correct_leaf_ids"].apply(len).sum()
        total_kept_wrong = group["kept_wrong_leaf_ids"].apply(len).sum()
        total_gold = group["gold_child_ids"].apply(len).sum()
        
        results.append({
            "system": sys_name,
            "rescued": total_rescued,
            "harmful_drift": total_harmful,
            "kept_correct": total_kept_correct,
            "kept_wrong": total_kept_wrong,
            "total_gold": total_gold,
            "rescue_rate": total_rescued / max(1, total_gold),
            "harmful_rate": total_harmful / max(1, total_gold + total_harmful),
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path.with_suffix(".csv"), index=False)
    
    md = "| System | Rescued | Harmful Drift | Kept Correct | Kept Wrong | Rescue Rate | Harmful Rate |\n"
    md += "|---|---:|---:|---:|---:|---:|---:|\n"
    for _, row in df_results.iterrows():
        md += f"| {row['system']} | {row['rescued']} | {row['harmful_drift']} | "
        md += f"{row['kept_correct']} | {row['kept_wrong']} | "
        md += f"{row['rescue_rate']:.4f} | {row['harmful_rate']:.4f} |\n"
    
    output_path.write_text(md)
    print(f"Generated {output_path}")


def generate_significance_table(artifact_dir: Path, output_path: Path):
    """Generate significance comparison table."""
    main_results_path = artifact_dir / "main_results.json"
    if not main_results_path.exists():
        return
    
    with open(main_results_path) as f:
        data = json.load(f)
    
    comparisons = data.get("comparisons_vs_flat", {})
    
    md = "| System | p-value vs Flat | 95% CI for Difference |\n"
    md += "|---|---:|---|\n"
    for sys_name, comp in comparisons.items():
        p = comp.get("official_qasper_evidence_f1_p_vs_flat", 0)
        ci = comp.get("official_qasper_evidence_f1_diff_cluster_ci", [0, 0])
        md += f"| {sys_name} | {p:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |\n"
    
    output_path.write_text(md)
    print(f"Generated {output_path}")


def generate_latency_plot(artifact_dir: Path, output_path: Path):
    """Generate latency vs performance plot."""
    df = load_artifacts(artifact_dir)
    if df.empty:
        return

    summary = df.groupby("system").agg({
        "official_qasper_evidence_f1": "mean",
        "latency_ms": "mean",
        "context_tokens": "mean",
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Evidence F1 vs Latency
    sns.scatterplot(data=summary, x="latency_ms", y="official_qasper_evidence_f1", 
                    hue="system", s=100, ax=axes[0])
    for _, row in summary.iterrows():
        axes[0].annotate(row["system"], (row["latency_ms"], row["official_qasper_evidence_f1"]))
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Official Evidence F1")
    axes[0].set_title("Evidence F1 vs Latency")
    axes[0].legend().remove()
    
    # Evidence F1 vs Context Tokens
    sns.scatterplot(data=summary, x="context_tokens", y="official_qasper_evidence_f1",
                    hue="system", s=100, ax=axes[1])
    for _, row in summary.iterrows():
        axes[1].annotate(row["system"], (row["context_tokens"], row["official_qasper_evidence_f1"]))
    axes[1].set_xlabel("Context Tokens")
    axes[1].set_ylabel("Official Evidence F1")
    axes[1].set_title("Evidence F1 vs Context Tokens")
    axes[1].legend().remove()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="data/artifacts/v7_dev_full")
    parser.add_argument("--output-dir", default="analysis/generated")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not artifact_dir.exists():
        print(f"Artifact directory not found: {artifact_dir}")
        return

    generate_main_table(artifact_dir, output_dir / "table1_main_results.md")
    generate_ablation_table(artifact_dir, output_dir / "table2_ablations.md")
    generate_rescue_harm_analysis(artifact_dir, output_dir / "table3_rescue_harm.md")
    generate_significance_table(artifact_dir, output_dir / "table4_significance.md")
    generate_latency_plot(artifact_dir, output_dir / "figure1_latency_performance.png")

    print(f"\nAll tables/figures generated in {output_dir}")


if __name__ == "__main__":
    main()